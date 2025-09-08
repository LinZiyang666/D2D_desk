#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_cpu_hybrid_pp.py

This is the main experiment script for training Qwen-0.6B using a custom 
Hybrid Pipeline Parallelism framework built on top of PiPPy.

- Stage 0 (rank 0): embedding + layers 0–13
- Stage 1 (ranks 1 & 2): layers 14–27 + norm + lm_head (with DDP)

Key Features:
- CPU-only execution with 3 ranks
- Manual microbatch scheduling via custom action list
- Used for testing correctness and loss behavior of the hybrid PP runtime
"""


import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from pipelining_source_code.schedules import _Action, _ComputationType

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

def create_pipeline_actions(upstream: int = None):

    # [stage],[rank],[id],[action type],[microbatch],[dest_rank],[upstream],[dependency]
    # Rank 0 (Stage 0)
    rank0_actions = [
        _Action(0, 0, 0, _ComputationType.FORWARD, (0,), None, None, None),
        _Action(0, 0, 1, _ComputationType.FULL_BACKWARD, (0,), None, None, None),   
        _Action(0, 0, 2, _ComputationType.ALL_REDUCE, None, None, upstream, None), 
    ]
    
    # Rank 1 (Stage 1)
    rank1_actions = [
        _Action(0, 1, 0, _ComputationType.FORWARD, (1,), None, None, None),
        _Action(0, 1, 1, _ComputationType.FULL_BACKWARD, (1,), None, None, None),
        _Action(0, 1, 2, _ComputationType.ALL_REDUCE, None, None, upstream, None),
    ]
    
    # Rank 2 (Stage 1)
    rank2_actions = [
        _Action(0, 2, 0, _ComputationType.FORWARD, (2,), None, None, None),
        _Action(0, 2, 1, _ComputationType.FULL_BACKWARD, (2,), None, None, None),
        _Action(0, 2, 2, _ComputationType.ALL_REDUCE, None, None, upstream, None),
    ]
    
    return {0: rank0_actions, 1: rank1_actions, 2: rank2_actions}

class Qwen3(nn.Module):
    """
    Single-module version that merges the previously split Part0/_MiddlePart/Part2.
    - Supports selecting the number of transformer layers (num_layers) or an explicit set of indices (keep_indices) at init time.
    - Keeps the same forward API as your split version: manually builds position_ids, pos_emb, and a causal attn_mask, and passes them into each layer.
    - Returns logits by default (suitable for training). If you need hidden states as well, adapt to return (logits, hidden).
    """
    def __init__(self, model, num_layers=None, keep_indices=None, force_tie_weight=True):
        """
        Args:
            model: A loaded Qwen3-0.6B HF model (has .model and .lm_head).
            num_layers (int|None): Use the first num_layers transformer blocks; if None, use all.
            keep_indices (List[int]|None): Explicit list of layer indices to keep (takes priority over num_layers).
            force_tie_weight (bool): If True, enforce lm_head.weight = embed_tokens.weight to keep tied weights.
        """
        super().__init__()
        base = model.model

        self.embed_tokens = base.embed_tokens           # token embedding
        self.norm = base.norm                           # final RMSNorm
        self.lm_head = model.lm_head                    # output head
        self.rotary_emb = base.rotary_emb               # RoPE position embedding (same usage as your split code)

        total_layers = len(base.layers)                 # typically 28
        if keep_indices is not None:
            sel = list(keep_indices)
        elif num_layers is not None:
            assert 1 <= num_layers <= total_layers, f"num_layers must be in [1, {total_layers}]"
            sel = list(range(num_layers))
        else:
            sel = list(range(total_layers))

        # Directly reference selected blocks (weights and grads preserved)
        self.layers = nn.ModuleList([base.layers[i] for i in sel])

        # Optionally re-tie lm_head and embeddings (some implementations already tie; this ensures it)
        if force_tie_weight and hasattr(self.lm_head, "weight") and hasattr(self.embed_tokens, "weight"):
            try:
                self.lm_head.weight = self.embed_tokens.weight
            except Exception:
                # Some implementations may forbid rebinding; safe to ignore
                pass

    @torch.no_grad()
    def _build_causal_mask(self, bsz, seqlen, device, dtype=torch.float32):
        # Shape (B, 1, T, T). Upper triangle is -inf, elsewhere 0.
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seqlen, seqlen).contiguous()
        return mask

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (B, T) token IDs.
            attention_mask: None / (T, T) / (B, 1, T, T).
        Returns:
            logits: (B, T, vocab_size)
        """
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        # Build position ids and token embeddings
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        hidden = self.embed_tokens(input_ids)

        # RoPE position embedding 
        cos, sin = self.rotary_emb(hidden, position_ids)

        # Attention mask handling
        if attention_mask is None:
            attn_mask = self._build_causal_mask(bsz, seqlen, device, dtype=torch.float32)
        elif attention_mask.dim() == 2:
            attn_mask = attention_mask.to(device)
            if attn_mask.dtype != torch.float32:
                attn_mask = attn_mask.to(torch.float32)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()
        else:
            attn_mask = attention_mask.to(device)
            if not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()

        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin),
                output_attentions=False,
                use_cache=False
            )[0]

        # Final normalization and output head
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits
    
parser = argparse.ArgumentParser()
parser.add_argument("--train_steps", type=int, default=None,
                    help="The total number of steps for training. If omitted, run the entire DataLoader.")
parser.add_argument("--batch_size", type=int,
                    default=int(os.getenv("BATCH_SIZE", 16)),
                    help="It should be the mutiple of 3. The batch size of each rank (the environment variable BATCH_SIZE can be overridden)")
parser.add_argument("--sudo_pass", default=os.getenv("SUDO_PASS"),
                    help='Write the password of root')
parser.add_argument("--middle_layers", type=int, default=6,
                    help='The number of layers in the middle of the model')
parser.add_argument("--upstream", type=int, default=None,
                    help='The upstream bandwidth')
args = parser.parse_args()
def main():

    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    name = "Qwen/Qwen3-0.6B-Base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    full = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
    middle_layers = args.middle_layers

    dp_group = dist.new_group(ranks=[0, 1, 2])
    stage_mod = Qwen3(full,num_layers=middle_layers)
    stage_mod.to(device)
    
    #Using DDP as the data parallelism component of our frame
    stage_mod = DDP(
        stage_mod,
        device_ids=None,        # CPU don' use device_ids
        process_group=dp_group, # Used for local dp
        find_unused_parameters=False,
        init_sync = False,
    )
    
    # #A hook for detailed output of allreduce
    # def timing_hook(state, bucket: dist.GradBucket):
    #     """
    #     Put a dict in the state:
    #         {"pg": dp_group, "use_cuda_event": True/False}
    #     """
    #     pg = state["pg"]
    #     use_evt = bucket.buffer().is_cuda and state.get("use_cuda_event", True)
    #     if use_evt:                    # GPU, timing with cudaEvent
    #         start_evt = torch.cuda.Event(enable_timing=True)
    #         end_evt   = torch.cuda.Event(enable_timing=True)
    #         start_evt.record()
    #     else:                          # CPU or not wanting to use Event
    #         t0 = time.perf_counter()
    #      # ------- key: specify group=pg ----------
    #     work = dist.all_reduce(bucket.buffer(), group=pg, async_op=True)
    #     def _callback(fut):
    #          # timing
    #         if use_evt:
    #             end_evt.record()
    #             end_evt.synchronize()
    #             elapsed_ms = start_evt.elapsed_time(end_evt)
    #         else:
    #             elapsed_ms = (time.perf_counter() - t0) * 1e3
    #         print(f"[Rank {dist.get_rank()}] Bucket {bucket.index()} "
    #             f"all‑reduce took {elapsed_ms:.3f} ms")
    #         # Be consistent with the default behavior of DDP: take the average
    #         bucket.buffer().div_(pg.size())
    #         return bucket.buffer()
    #      # Return the Future containing _callback
    #     return work.get_future().then(_callback)

    # stage_mod.register_comm_hook(state={"pg": dp_group}, hook=timing_hook)

    stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,  # Used for world pp 
                            prev_group=None, this_group=[0, 1, 2], next_group=None)
    
    del full                        
    import gc; gc.collect()

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    block = 128
    def tok_fn(ex): 
        return tok(ex["text"], return_attention_mask=False)
    def grp_fn(ex):
        concat = sum(ex["input_ids"], [])
        tot = len(concat) // block * block
        ids = [concat[i:i+block] for i in range(0, tot, block)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}
    
    ds = (raw.map(tok_fn, batched=True, remove_columns=["text"])
              .map(grp_fn, batched=True))
    ds.set_format("torch", columns=["input_ids", "labels"])
    
    batch_size = args.batch_size
    microbatch_num = 3
    
    if rank == 0:
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    def loss_fn(output, target):
        if output is None or target is None:
            return None

        vocab_size = output.size(-1)

        if output.dim() == 3:
            # output: (B, T, V), target: (B, T)
            output = output[:, :-1, :].reshape(-1, vocab_size)
            target = target[:, 1:].reshape(-1)
        elif output.dim() == 2:
            # output: (B*T, V), target: (B*T)
            output = output.reshape(-1, vocab_size)
            target = target.reshape(-1)
        else:
            raise ValueError(f"[loss_fn] Unexpected output shape: {output.shape}")

        return F.cross_entropy(output, target)


    
    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=microbatch_num, loss_fn=loss_fn, root_pass=args.sudo_pass)
    actions = create_pipeline_actions(upstream=args.upstream)
    sched._load_actions(actions, format="compute_comms")
    
    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)            
    prev_loss = None
    
    
    for epoch in range(1):
        if rank == 0:
            if args.train_steps is None:
                steps_tensor = torch.tensor(len(loader), device=device)
            else:
                steps_tensor = torch.tensor(args.train_steps, device=device)
            dist.broadcast(steps_tensor, 0)
            data_iter = iter(loader)
            print(f"Total training steps: {steps_tensor.item()}")
        else:
            steps_tensor = torch.tensor(0, device=device)
            dist.broadcast(steps_tensor, 0)

        total_steps = int(steps_tensor.item())

        if rank == 0:
            pbar = tqdm(total=int(total_steps), 
                       desc=f"Training Epoch {epoch+1}",
                       unit="step",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            start_time = time.time()
        
        for step in range(total_steps):
            step_start_time = time.time()
            opt.zero_grad(set_to_none=True)
            
            if rank == 0:
                batch = next(data_iter)
                inp = batch["input_ids"].to(device)
                tgt = batch["labels"].to(device)
                dist.broadcast(tgt, src=0)
                sched.step(inp, target=tgt)
            else:
               
                tgt = torch.empty(batch_size, block, dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)
                sched.step(target=tgt)
            
            if (step + 1) % 50 == 0:
                sched.timeline_rec.events.clear()
            opt.step()
            
            if rank == 0:
                step_time = time.time() - step_start_time
                tokens_processed = batch_size * block
                tokens_per_second = tokens_processed / step_time
                
                pbar.set_postfix({
                    'tokens/s': f'{tokens_per_second:.0f}',
                    'step_time': f'{step_time:.2f}s',
                    'lr': f'{opt.param_groups[0]["lr"]:.2e}'
                })
                pbar.update(1)
                
            cur_loss = getattr(sched, "last_step_loss", None)
            
            if cur_loss is not None:
                delta = (cur_loss - prev_loss) if prev_loss is not None else None

                print(f"[rank0] step {step+1} loss {cur_loss:.4f}"
                        + ("" if prev_loss is None else (" down" if cur_loss < prev_loss else " up" if cur_loss > prev_loss else " flat")))
                prev_loss = cur_loss
            
            
            
            dist.barrier()
        
        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {total_steps / total_time:.2f} steps/s")

    if rank == 0:
        print("\nMerging and saving model...")
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        merged = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        merged_state = merged.state_dict()
        
        part1_state = stage_mod.state_dict()
        for key in part1_state:
            if key.startswith("embed_tokens"):
                merged_state[f"model.{key}"] = part1_state[key]
            elif key.startswith("layers"):
                merged_state[f"model.{key}"] = part1_state[key]
            elif key.startswith("rotary_emb"):
                merged_state[f"model.{key}"] = part1_state[key]
        
        recv = [None]
        dist.broadcast_object_list(recv, src=1)
        part2_state = recv[0]
        
        for key in part2_state:
            if key.startswith("layers"):
                layer_idx = int(key.split(".")[1]) + 14
                new_key = f"model.layers.{layer_idx}" + key[key.find(".", 7):]
                merged_state[new_key] = part2_state[key]
            elif key == "norm.weight":
                merged_state["model.norm.weight"] = part2_state[key]
            elif key == "lm_head.weight":
                merged_state["lm_head.weight"] = part2_state[key]
                
        merged.load_state_dict(merged_state, strict=False)
        merged.save_pretrained("trained_qwen_pp")
        tok.save_pretrained("trained_qwen_pp")
        print("Saved to ./trained_qwen_pp")
    else:
        dist.broadcast_object_list([stage_mod.state_dict()], src=1)

    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()