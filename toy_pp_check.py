#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
toy_pp_check.py

This script is designed to verify the correctness of a custom Heterogeneous Hybrid Pipeline Parallelism 
(HHPP) framework built on top of PiPPy. It simulates a deterministic 2-stage pipeline model with data 
parallelism on the second stage, using handcrafted schedule actions and a fixed toy model to allow 
step-by-step inspection of forward and backward behavior.

Key Features:
- Stage 0 (rank 0): A simple Linear(2→2) layer
- Stage 1 (ranks 1 and 2): A shared Linear(2→1) layer, optionally wrapped with DDP
- Fully deterministic weights, inputs, and expected SGD updates
- Manual definition of pipeline computation and communication schedule
- One forward-backward-optimizer step to validate:
    - Gradients are correctly computed and propagated
    - Weights are updated as expected
    - Runtime behavior matches analytically computed results

Usage:
    torchrun --nproc-per-node=3 toy_pp_check.py [--use_ddp]

Requirements:
- Run with exactly 3 ranks
- Torch distributed backend (e.g., gloo)
- All operations performed on CPU for reproducibility

This script is a core utility for validating the runtime and correctness of hybrid pipeline 
parallelism implementations with non-trivial scheduling and inter-rank communication.
"""

import os
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from pipelining_source_code.schedules import _Action, _ComputationType

# ----------------------------
# Config: deterministic setup
# ----------------------------
IN_DIM = 2
HID_DIM = 2
OUT_DIM = 1
MB = 2             # 2 microbatches, one per DP rank
LR = 0.1
DTYPE = torch.float32
DEVICE = torch.device("cpu")

# Data (2 samples = 2 microbatches)
# x0 -> rank1, x1 -> rank2 (by schedule)
X = torch.tensor([[1.0,  2.0],
                  [0.0, -1.0]], dtype=DTYPE)
Y = torch.tensor([[ 1.0],
                  [-1.5]],      dtype=DTYPE)

# Initial weights
# Stage0 W1: identity (2x2)
W1_INIT = torch.tensor([[1.0, 0.0],
                        [0.0, 1.0]], dtype=DTYPE)
# Stage1 W2: [0.5, -1.0] (1x2)
W2_INIT = torch.tensor([[0.5, -1.0]], dtype=DTYPE)


# ----------------------------
# Tiny stage modules
# ----------------------------
class ToyPart1(nn.Module):
    # Stage 0: Linear(2->2), no bias
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(IN_DIM, HID_DIM, bias=False)
        with torch.no_grad():
            self.W1.weight.copy_(W1_INIT)

    def forward(self, x):
        # x: (B, 2) -> h: (B, 2)
        return (self.W1(x),)  # tuple to match pipeline expectations


class ToyPart2(nn.Module):
    # Stage 1: Linear(2->1), no bias
    def __init__(self):
        super().__init__()
        self.W2 = nn.Linear(HID_DIM, OUT_DIM, bias=False)
        with torch.no_grad():
            self.W2.weight.copy_(W2_INIT)   # Explicitly initialized to the same weight

    def forward(self, h):
        # h: (B, 2) -> yhat: (B, 1)
        return self.W2(h)


# ----------------------------
# Schedule (compute+comms)
# 2 microbatches total: mb0 -> rank1, mb1 -> rank2
# _Action：
# [stage],[rank],[id],[action type],[microbatch],[dest_rank],[upstream],[dependency]
# ----------------------------
def create_toy_actions():
    A = _ComputationType
    # Rank 0 (Stage 0)
    r0 = [
        _Action(0, 0, 0, A.FORWARD,       (0,1), None, None, None),
        _Action(0, 0, 1, A.SEND_F,        (0,),  1,    None, None),
        _Action(0, 0, 2, A.SEND_F,        (1,),  2,    None, None),
        _Action(0, 0, 3, A.RECV_B,        (0,),  1,    None, None),
        _Action(0, 0, 4, A.RECV_B,        (1,),  2,    None, None),

        _Action(0, 0, 5, A.FULL_BACKWARD, (0,1), None, None, None),
    ]

    # Rank 1 (Stage 1, DP member)
    r1 = [
        _Action(1, 1, 0, A.RECV_F,        (0,),  0,    None, None),
        _Action(1, 1, 1, A.FORWARD,       (0,),  None, None, None),
        _Action(1, 1, 2, A.FULL_BACKWARD, (0,),  None, None, None),
        _Action(1, 1, 3, A.SEND_B,        (0,),  0,    None, None),
        _Action(1, 1, 4, A.ALL_REDUCE,    None,  None, None, None),  # average over DP group
    ]

    # Rank 2 (Stage 1, DP member)
    r2 = [
        _Action(1, 2, 0, A.RECV_F,        (1,),  0,    None, None),
        _Action(1, 2, 1, A.FORWARD,       (1,),  None, None, None),
        _Action(1, 2, 2, A.FULL_BACKWARD, (1,),  None, None, None),
        _Action(1, 2, 3, A.SEND_B,        (1,),  0,    None, None),
        _Action(1, 2, 4, A.ALL_REDUCE,    None,  None, None, None),
    ]
    return {0: r0, 1: r1, 2: r2}


# ----------------------------
# Expected values under SGD
# ----------------------------
@dataclass
class ExpectedStep:
    loss_mean: float
    W1_next: torch.Tensor  # (2,2)
    W2_next: torch.Tensor  # (1,2)

def compute_expected_one_step():
    """
    Equivalent to training on the *mean loss* over the two samples (DP averaging).
    """
    W1 = W1_INIT.clone()
    W2 = W2_INIT.clone()

    x = X.clone()
    y = Y.clone()

    # forward
    h = x @ W1.t()              # (2,2)
    yhat = h @ W2.t()           # (2,1)
    err = (yhat - y)            # (2,1)

    # MSE(mean over all elems): here 2 scalars -> mean of two squared errors
    loss = (err.pow(2).mean()).item()

    # grads for mean loss:
    # dL/dyhat = 2*(yhat - y) / 2 = (yhat - y)
    dL_dyhat = err.clone()      # (2,1)

    # W2 grad = (dL/dyhat)^T @ h  => (1,2)
    gW2 = dL_dyhat.t() @ h

    # h grad = dL/dyhat @ W2     => (2,2)
    dL_dh = dL_dyhat @ W2

    # W1 grad = (dL/dh)^T @ x    => (2,2)
    gW1 = dL_dh.t() @ x

    # SGD update
    W2_next = W2 - LR * gW2
    W1_next = W1 - LR * gW1

    return ExpectedStep(loss_mean=loss, W1_next=W1_next, W2_next=W2_next)


# ----------------------------
# Loss used by runtime (last stage)
# ----------------------------
def mse_loss(output, target):
    if output is None or target is None:
        return None
    return torch.mean((output - target) ** 2)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ddp", action="store_true",
                        help="Wrap Stage-1 with DDP to exercise DDP-trigger allreduce path. Otherwise use manual allreduce.")
    args = parser.parse_args()

    torch.manual_seed(0)
    dist.init_process_group("gloo", init_method="env://")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 3, "Run with exactly 3 ranks (rank0=stage0, rank1/2=stage1 DP)."
    
    # ---------- Build stage modules ----------
    if rank == 0:
        stage_mod = ToyPart1().to(DEVICE)
        stage = PipelineStage_with_mutiple_ranks(
            stage_mod, stage_index=0, num_stages=3, device=DEVICE,
            group=dist.group.WORLD, prev_group=None, this_group=[0], next_group=[1,2]
        )
        params_for_opt = stage_mod.parameters()
    else:
        sub = ToyPart2().to(DEVICE)
        if args.use_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            sub = DDP(sub, device_ids=None,
                      broadcast_buffers=False, find_unused_parameters=False,
                      init_process_group=None, static_graph=False, 
                      init_sync=False)
        stage = PipelineStage_with_mutiple_ranks(
            sub, stage_index=1, num_stages=3, device=DEVICE,
            group=dist.group.WORLD, prev_group=[0], this_group=[1,2], next_group=None
        )
        params_for_opt = sub.parameters()


    # ---------- Assemble schedule ----------
    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=MB, loss_fn=mse_loss)
    sched._load_actions(create_toy_actions(), format="compute_comms")

    opt = optim.SGD(params_for_opt, lr=LR)

    # ---------- One deterministic step ----------
    dist.barrier()
    if rank == 0:
        inp = X.to(DEVICE)     # (2,2)
        tgt = Y.to(DEVICE)     # (2,1)
        #Let all ranks have the same target (which will be used at the end)
        dist.broadcast(tgt, src=0)
        #Print the parameters and gradients of rank0 before step (expect that grad has not yet appeared at this moment)
        with torch.no_grad():
            w_sum_before = sum(p.detach().float().sum().item() for p in stage_mod.parameters())
        print(f"[rank0:before-step] param sum = {w_sum_before:.8f}")

        sched.step(inp, target=tgt)

        # Print the gradient overview after reverse (before step)
        grad_info = []
        for n, p in stage_mod.named_parameters():
            gmax = None if (p.grad is None) else p.grad.detach().abs().max().item()
            grad_info.append((n, "None" if gmax is None else f"{gmax:.8f}"))
        print(f"[rank0:after-backward-before-opt] grad max per param = {grad_info}")
    else:
        tgt = torch.empty_like(Y, device=DEVICE)
        dist.broadcast(tgt, src=0)
        sched.step(target=tgt)

    # All ranks require step (Stage0/Stage1 update their respective parameters respectively)
    opt.step()

    # rank0 prints the parameters and changes again
    if rank == 0:
        with torch.no_grad():
            w_sum_after = sum(p.detach().float().sum().item() for p in stage_mod.parameters())
        print(f"[rank0:after-opt] param sum = {w_sum_after:.8f}")

    dist.barrier()

    # ---------- Collect results & check ----------
    # Collect Stage-1 weights from rank1 to rank0 (no broadcasting required, only collect and print)
    stage1_state = [None]
    if rank == 1:
        state = stage.submod.state_dict() if not args.use_ddp else stage.submod.module.state_dict()
        stage1_state[0] = {k: v.detach().cpu() for k, v in state.items()}
    dist.broadcast_object_list(stage1_state, src=1)

    # Only perform validation in rank0
    if rank == 0:
        exp = compute_expected_one_step()

        step_loss = getattr(sched, "last_step_loss", None)
        print(f"[check] expected loss={exp.loss_mean:.8f}, runtime loss={step_loss:.8f}")

        W1_actual = stage.submod.W1.weight.detach().cpu()
        maxdiff_W1 = (W1_actual - exp.W1_next).abs().max().item()
        print(f"[check] Stage0.W1 max|diff|={maxdiff_W1:.8e}")

        W2_actual = stage1_state[0]["W2.weight"]
        maxdiff_W2 = (W2_actual - exp.W2_next).abs().max().item()
        print(f"[check] Stage1.W2 max|diff|={maxdiff_W2:.8e}")

        tol = 1e-6
        ok = (abs(step_loss - exp.loss_mean) < 1e-6) and (maxdiff_W1 < tol) and (maxdiff_W2 < tol)
        print("[RESULT]", "PASS" if ok else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
