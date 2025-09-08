# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Union

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.fx.node import Argument, map_aggregate
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map_only

from pipelining_source_code.stage import PipelineStage, InputInfo, _RecvInfo, _RootArgPlaceholder, _normalize_model_output_as_tuple
from pipelining_source_code._debug import map_debug_info
from pipelining_source_code._utils import flatten_args
from pipelining_source_code._backward import stage_backward, stage_backward_input, stage_backward_weight
logger = logging.getLogger(__name__)

class PipelineStage_with_mutiple_ranks(PipelineStage):
    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
        prev_group: list[int] = None, 
        this_group: list[int] = None, 
        next_group: list[int] = None
    ):
        self.prev_group = prev_group          # list[int] | None
        self.this_group = this_group          # list[int]
        self.next_group = next_group          # list[int] | None
        
        self.dp_group = dist.new_group(ranks=this_group)
        self.leader   = min(this_group)

        self.is_leader = dist.get_rank() == self.leader
        
        super().__init__(submodule, stage_index, num_stages, device, input_args, output_args, group, dw_builder)
        
        
        
    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.prev_group is None

    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.next_group is None
    
    def _shape_inference(self, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        assert args is not None

        # --- Common ---
        dp_size = dist.get_world_size(self.dp_group)
        is_multi_dp = dp_size > 1

        # === 1) Handle inputs ===
        if self.prev_group is None:
            # First stage (pure DP or "first stage + DP")
            if is_multi_dp:
                if self.is_leader:
                    meta_args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
                    # Leader must also participate in the same broadcast
                    dist.broadcast_object_list([meta_args], src=self.leader, group=self.dp_group)
                    args = meta_args
                else:
                    buf = [None]
                    dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                    args = buf[0]
            else:
                # dp_size == 1, no collectives needed
                args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
        else:
            # Non-first stage: receive from previous stage's leader, then DP-broadcast locally
            prev_leader = min(self.prev_group)
            if self.is_leader:
                buf = [None]
                dist.recv_object_list(buf, src=prev_leader, group=dist.group.WORLD)  # world group
                args = buf[0]
                # Safety: ensure meta in case upstream sent real tensors by mistake
                args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
                if is_multi_dp:
                    dist.broadcast_object_list([args], src=self.leader, group=self.dp_group)
            else:
                if is_multi_dp:
                    buf = [None]
                    dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                    args = buf[0]
                else:
                    # dp_size == 1 and not leader cannot happen
                    raise RuntimeError("Unexpected non-leader with dp_size==1")

        # Basic sanity
        assert isinstance(args, tuple), f"shape-infer args must be tuple, got {type(args)}"

        # === 2) Cache inputs & run a dummy forward on zeros to compute outputs' shapes ===
        self.inputs_meta = args
        args = tree_map_only(torch.Tensor, lambda x: torch.zeros_like(x, device=self.device), args)

        with torch.no_grad():
            outputs = self.submod(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        outputs_meta = tuple(tree_map_only(torch.Tensor, lambda x: x.to("meta"), outputs))
        self._configure_outputs_meta(outputs_meta)

        # === 3) Send to next stage (only leader), and DP-broadcast locally for consistency ===
        if self.next_group is not None:
            next_leader = min(self.next_group)
            if self.is_leader:
                dist.send_object_list([outputs_meta], dst=next_leader, group=dist.group.WORLD)  # world group

        if is_multi_dp:
            dist.broadcast_object_list([outputs_meta], src=self.leader, group=self.dp_group)

        return outputs_meta

    
    def _get_recv_ops(
        self,
        recv_infos: tuple[InputInfo, ...],
        rank: int, 
        dest_rank: int
    ) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: list[dist.P2POp] = []
        for info in recv_infos:
            if not isinstance(info, _RecvInfo):
                continue

            peer_rank = dest_rank
            peer_global_rank = (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )
            ops.append(
                dist.P2POp(dist.irecv, info.buffer, peer_global_rank, self.group)
            )

        return ops
    
    def get_fwd_recv_ops(self, fwd_chunk_id: int, rank: int, dest_rank: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info[fwd_chunk_id]

        return self._get_recv_ops(recv_infos, rank, dest_rank)

    def get_bwd_recv_ops(self, bwd_chunk_id: int, rank: int, dest_rank: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        return self._get_recv_ops(recv_infos, rank, dest_rank)

    def get_fwd_send_ops(self, fwd_chunk_id: int, rank: int, dest_rank: int) -> list[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        """
        output_tuple, _ = self.fwd_cache[fwd_chunk_id]

        ops: list[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                logger.debug(
                    "%s Sending tensor to Stage %s: %s",
                    self.log_prefix,
                    dst,
                    out.size(),
                )
                peer_rank = dest_rank
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(dist.P2POp(dist.isend, out, peer_global_rank, self.group))

        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int, rank: int, dest_rank: int) -> list[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        if not self.has_backward or self.is_first:
            return []

        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        ops: list[dist.P2POp] = []
        grads_input = self.bwd_cache.pop(bwd_chunk_id)
        self.fwd_cache.pop(bwd_chunk_id, None)
        
        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
            # 跳过 None 的梯度接收阶段（对应于整数类型的tensor）
            if grad_recv_stage is None:
                continue
                
            if isinstance(grad, torch.Tensor):
                # 额外检查：只发送浮点类型的梯度
                if not (grad.is_floating_point() or torch.is_complex(grad)):
                    print(f"[{dist.get_rank()}] Skipping non-floating grad send: dtype={grad.dtype}")
                    continue
                    
                logger.debug(
                    "%s Sending gradient to Stage %s: %s",
                    self.log_prefix,
                    grad_recv_stage,
                    grad.size(),
                )
                peer_rank = dest_rank
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(dist.P2POp(dist.isend, grad, peer_global_rank, self.group))
            elif grad is not None:
                raise RuntimeError(
                    f"[{self.stage_index}] expecting a gradient tensor for an input "
                    f"coming from stage {grad_recv_stage}, but got {type(grad)}"
                )
        
        return ops
    
    def _execute_allreduce(self):
        """
        Execute allreduce for gradients across the data parallel group.
        This should be called after all microbatches have been processed.
        """
        if not hasattr(self, 'dp_group') or self.dp_group is None:
            return
            
        # Check if the submodule is wrapped with DDP
        if isinstance(self.submod, DistributedDataParallel):
            # If using DDP, we need to manually trigger allreduce
            # since we've been using no_sync() during backward
            self._trigger_ddp_allreduce()
        else:
            # If not using DDP, manually allreduce all gradients
            self._manual_allreduce_gradients()
    
    def _trigger_ddp_allreduce(self):
        """
        Trigger DDP's allreduce mechanism manually.
        This is used when we've been accumulating gradients with no_sync().
        """
        if not isinstance(self.submod, DistributedDataParallel):
            return
            
        # Method 1: Use prepare_for_backward and trigger with a dummy backward
        # This is the approach used in the original paste.txt
        
        # Collect all output tensors from forward passes
        # Since we don't have access to the actual outputs here, 
        # we create a dummy tensor that depends on all parameters with gradients
        dummy_tensors = []
        for param in self.submod.parameters():
            if param.requires_grad and param.grad is not None:
                # Create a dummy tensor that depends on this parameter
                # Using a very small value to not affect actual gradients
                dummy = param.sum() * 0.0
                if dummy.requires_grad:
                    dummy_tensors.append(dummy)
        
        if not dummy_tensors:
            return
            
        # Prepare reducer for backward
        self.submod.reducer.prepare_for_backward(dummy_tensors)
        
        # Temporarily enable gradient synchronization
        original_sync = self.submod.require_backward_grad_sync
        self.submod.require_backward_grad_sync = True
        
        # Trigger backward to activate allreduce
        if dummy_tensors:
            # Sum all dummy tensors and backward
            total_dummy = sum(dummy_tensors)
            total_dummy.backward()
        
        # Restore original sync setting
        self.submod.require_backward_grad_sync = original_sync
    
    def _manual_allreduce_gradients(self):
        """
        Manually perform allreduce on all gradients when not using DDP.
        """
        if self.dp_group is None:
            return
            
        world_size = dist.get_world_size(self.dp_group)
        
        # Collect all parameters with gradients
        params_with_grad = []
        for param in self.submod.parameters():
            if param.grad is not None:
                params_with_grad.append(param)
        
        if not params_with_grad:
            return
        
        # Perform allreduce on each gradient
        # Can be optimized by batching small gradients together
        for param in params_with_grad:
            # Allreduce the gradient
            dist.all_reduce(
                param.grad.data,
                op=dist.ReduceOp.SUM,
                group=self.dp_group
            )
            # Average the gradient
            param.grad.data /= world_size
    
    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
        pack_size: int = 1
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """


        def _is_first_stage_dp(self):
            try:
                dp_size = dist.get_world_size(self.dp_group)
            except Exception:
                dp_size = 1
            return (self.prev_group is None) and (dp_size > 1)

        need_rt_bcast = _is_first_stage_dp(self)

        composite_args = None

        if need_rt_bcast:
            if self.is_leader:
                if not args or len(args) == 0:
                    raise RuntimeError(
                        f"[rank{dist.get_rank()}] First-stage leader got empty args at "
                        f"fwd_chunk_id={fwd_chunk_id}. Scheduler must pass root inputs to leader."
                    )
                dist.broadcast_object_list([args], src=self.leader, group=self.dp_group)
                composite_args = args
            else:
                buf = [None]
                dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                composite_args = buf[0]

        else:
            if args:
                composite_args = args
            else:
                composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        if composite_args is None or len(composite_args) == 0:
            raise RuntimeError(
                f"[rank{dist.get_rank()}] Empty composite_args after dispatch at "
                f"stage={self.stage_index}, fwd_chunk_id={fwd_chunk_id}, "
                f"is_first={self.prev_group is None}."
            )


        composite_kwargs = kwargs or {}

        if (
            pack_size > 1
            and composite_args
            and isinstance(composite_args[0], torch.Tensor)
        ):
            mb_bs = composite_args[0].shape[0] // pack_size
            args_for_val = tuple(
                (t[:mb_bs] if isinstance(t, torch.Tensor) else t)
                for t in composite_args
            )
            kwargs_for_val = {
                k: (v[:mb_bs] if isinstance(v, torch.Tensor) else v)
                for k, v in composite_kwargs.items()
            }
        else:
            args_for_val = composite_args
            kwargs_for_val = composite_kwargs

        self._validate_fwd_input(args_for_val, kwargs_for_val)

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # See [Note: pipeline model output type]
        output_tuple = _normalize_model_output_as_tuple(output)

        # Prepare for final output merge or reduction
        # Output chunks is only used for the last stage since we only merge the output of the last stage
        if self.is_last:
            self.output_chunks.append(output)

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            map_debug_info(output),
        )
        if pack_size > 1 and isinstance(output_tuple, tuple) and len(output_tuple):
            outputs_for_val = tuple(
                t[:mb_bs] if isinstance(t, torch.Tensor) else t
                for t in output_tuple
            )
        elif pack_size > 1 and isinstance(output_tuple, torch.Tensor):
            outputs_for_val = output_tuple[:mb_bs]
        else:
            outputs_for_val = output_tuple

        self._validate_fwd_outputs(outputs_for_val)

        # We return the original user-provied output, not normalized to tuple.
        # See [Note: pipeline model output type]
        return output

    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
        retain_graph_for_packed_mbs: bool = False,
    ) -> tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]]:
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]],
        ]:
            if backward_type == "full":
                return lambda: (
                    stage_backward(
                        bwd_kwargs["stage_output"],
                        bwd_kwargs["output_grads"],
                        bwd_kwargs["input_values"],
                        retain_graph_for_packed_mbs=retain_graph_for_packed_mbs
                    ),
                    None,
                )
            elif backward_type == "input":
                return lambda: stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
            elif backward_type == "weight":
                return lambda: (
                    stage_backward_weight(
                        self.submod.parameters(), bwd_kwargs["param_groups"]
                    ),
                    None,
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

        # If submod is wrapped by DDP
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward:
                # Last chunk, prepare for gradient reduction
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                result = perform_backward(backward_type)()
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    result = perform_backward(backward_type)()
        # If submod is a FSDP module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(False)
            self.submod.set_reshard_after_backward(False)
            self.submod.set_requires_gradient_sync(False)
            result = perform_backward(backward_type)()
            if last_backward:
                # Manually call post backward for FSDP
                def run_post_backward(fsdp_module: FSDPModule) -> None:
                    fsdp_module.set_is_last_backward(True)
                    fsdp_module.set_reshard_after_backward(True)
                    fsdp_module.set_requires_gradient_sync(True)
                    fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]
                    for state in fsdp_state._state_ctx.all_states:
                        if state._fsdp_param_group:
                            state._fsdp_param_group.post_backward()

                    # it would be much better if pipelining backward invoked .backward so autograd hooks
                    # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
                    # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
                    fsdp_state._root_post_backward_final_callback()

                run_post_backward(self.submod)

        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups
    
    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
        retain_graph_for_packed_mbs: bool = False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
             # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)

            # === NEW (Plan B): average upstream grads across the next stage's DP replicas ===
            # If the next stage has N data-parallel members (len(self.next_group) == N),
            # each sender contributed a local gradient. To keep "mean loss" semantics,
            # divide by N on the receiver before autograd.backward.
            grads_output = self._avg_next_stage_grads(grads_output)
            # === END NEW ===

            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: tuple[Optional[torch.Tensor], ...] = ()

        # Custom backward function
        if self.dw_builder:
            # TODO: We may want to change our semantics so we are allowed to ignore
            # the 'dw_builder' and call full_backward directly when it is a full_backward op.
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
                last_backward=last_backward,
                retain_graph_for_packed_mbs = retain_graph_for_packed_mbs
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward,
                    retain_graph_for_packed_mbs = retain_graph_for_packed_mbs
                )
            else:
                param_groups: list[dict[str, Any]] | None = None
                # Skip the backward for the first stage since we will perform the weight update with
                # autograd.backward in backward_weight_one_chunk
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                    # perform the partial backwards for the inputs with a custom backward function
                    # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs, last_backward=last_backward
                    )

                # TODO: we dont need to save this, add to dw_runner?
                self.backward_state[bwd_chunk_id] = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                # Save a placeholder for the dw_runner
                self.dw_runner[bwd_chunk_id] = lambda: None

        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        grads_input = tuple(
            (g if dst is not None and isinstance(g, torch.Tensor) else None)
            for g, dst in zip(grads_input, self.grad_send_info)
        )
        
        self.bwd_cache[bwd_chunk_id] = grads_input

        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                if not t._is_view():  # views are not detachable in-place
                    t.detach_()

        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)
        
    
    def _avg_next_stage_grads(self, grads):
        """
        Average upstream gradients received from the *next* stage across its DP replicas.

        Why:
        - If the next stage is data-parallel over N ranks (len(self.next_group) == N),
            each rank sends its *local* gradient dL/d(out). Summing those locally on this stage
            would implement "sum loss" semantics. To keep the common "mean loss" semantics while
            keeping LR unchanged, we should divide the received gradients by N before calling
            autograd.backward on this stage.

        Behavior:
        - No-op when there is no next stage (last stage) or N <= 1.
        - Supports Tensor / list / tuple / None (None is passed through).
        - In-place scaling is used to avoid extra allocations.
        """
        import torch
        import torch.distributed as dist

        fan_in = len(self.next_group) if self.next_group is not None else 1
        if grads is None or fan_in <= 1:
            return grads

        # Helper to scale a single item
        def _scale_one(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                # in-place division; safe for gradient tensors
                return x.div_(fan_in)
            return x  # for unexpected types we leave as-is

        # Compute a small debug metric before scaling (leader only) 
        pre_max = None
        try:
            if self.is_leader:
                if isinstance(grads, (list, tuple)) and len(grads) > 0 and isinstance(grads[0], torch.Tensor):
                    pre_max = grads[0].abs().max().item()
                elif isinstance(grads, torch.Tensor):
                    pre_max = grads.abs().max().item()
        except Exception:
            pre_max = None

        # Apply scaling to common containers
        if isinstance(grads, list):
            out = [_scale_one(g) for g in grads]
        elif isinstance(grads, tuple):
            out = tuple(_scale_one(g) for g in grads)
        else:
            out = _scale_one(grads)

        # Debug print after scaling (leader only)
        if self.is_leader and pre_max is not None:
            try:
                post_max = (
                    (out[0].abs().max().item() if isinstance(out, (list, tuple)) else out.abs().max().item())
                )
                # print(
                #     f"[{dist.get_rank()}] AVG upstream grads by 1/{fan_in} "
                #     f"(stage_idx={self.stage_index}) -> max|g| {pre_max:.6f} -> {post_max:.6f}"
                # )
            except Exception:
                pass

        return out
    
    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple:
        grad_recv_info_list = []
        outputs_meta = self.get_outputs_meta()

        if not self.is_last:
            for idx, dst_list in act_send_info.items():
                dst = dst_list[0]
                meta = outputs_meta[idx] if idx < len(outputs_meta) else None

                if meta is None or not torch.is_tensor(meta):
                    grad_recv_info_list.append(None)
                    continue
                
                if not (meta.is_floating_point() or torch.is_complex(meta)):
                    grad_recv_info_list.append(None)
                    continue

                buffer = _make_tensor_from_meta(meta, self.device)
                grad_recv_info_list.append(
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{dst}",
                        dst,
                        buffer,
                    )
                )

        return tuple(grad_recv_info_list)
    
    def _create_grad_send_info(
        self,
        args_recv_info: tuple,
    ) -> list[Optional[int]]:
        grad_send_info: list[Optional[int]] = []

        def map_recv_to_send(a):
            if isinstance(a, _RecvInfo) and getattr(a, "buffer", None) is not None:
                if isinstance(a.buffer, torch.Tensor) and (
                    a.buffer.is_floating_point() or torch.is_complex(a.buffer)
                ):
                    grad_send_info.append(a.source)
                    return a.source
            grad_send_info.append(None)
            return None

        map_aggregate(args_recv_info, map_recv_to_send)
        return grad_send_info
    
    def _map_tensor_from_recv_info(
        self,
        recv_infos: tuple[InputInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if info is None:
                return None
            elif isinstance(info, _RecvInfo):
                return info.buffer
            else:
                raise AssertionError(f"Expected _RecvInfo or None but got {type(info)}")

        return map_aggregate(cast(Argument, recv_infos), get_recv_tensor)

def _make_tensor_from_meta(
    example: Union[torch.Tensor, FakeTensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a real tensor from a tensor.
    """
    return torch.empty(
        example.size(),
        dtype=example.dtype,
        layout=example.layout,
        device=device,
    )


