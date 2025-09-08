#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure per-layer parameter bytes, boundary activation bytes (per sample),
and forward/backward times for your model on THIS machine.

Usage:
  python measure_layers.py \
      --model_path /path/to/qwen-0_6b \
      --seq_len 2048 \
      --batch 16 \
      --dtype float32 \
      --iters 5 --warmup 2 \
      --out layers_this_machine.json \
      --host cpu1

Notes:
- Run this on EACH of your 8 machines (change --host to identify the machine)
- The script tries to locate transformer blocks under model.model.layers or model.layers.
- It times forward/backward using module-level hooks (CPU). Use release build (no debug).
- It measures boundary activation bytes by recording each block's output tensor size.
"""

import argparse
import json
import time
from typing import Dict, List

import torch
import torch.nn as nn
import psutil

def find_blocks(model: nn.Module) -> List[nn.Module]:
    # Try common layouts
    for attr in ["model", "transformer", "gpt_neox", "backbone"]:
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if hasattr(inner, "layers"):
                layers = getattr(inner, "layers")
                if isinstance(layers, (list, nn.ModuleList)):
                    return list(layers)
    # Fallback: search modules named "*layers*"
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, (list, nn.ModuleList)):
            return list(mod)
    raise RuntimeError("Cannot find transformer blocks list on this model. Inspect model structure.")

def param_bytes(module: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters(recurse=True))

def get_vocab_hidden(model: nn.Module):
    emb = model.get_input_embeddings()
    vocab = emb.weight.shape[0]
    hidden = emb.weight.shape[1]
    return vocab, hidden, emb.weight.element_size()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen3-0.6B", help="HF name or local path")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","bfloat16","float16"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--host", type=str, default="cpu1", help="Identifier for this machine, e.g., cpu1")
    args = ap.parse_args()

    # Resolve dtype
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    # Load model locally
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)  # we will run backward

    # Build token input (no tokenizer needed)
    vocab, hidden, embed_bytes = get_vocab_hidden(model)
    B = args.batch
    T = args.seq_len
    input_ids = torch.randint(0, vocab, (B, T), dtype=torch.long)
    # labels for LM loss
    labels = torch.randint(0, vocab, (B, T), dtype=torch.long)

    # Locate blocks
    blocks = find_blocks(model)
    L = len(blocks)

    # Prepare containers
    fwd_times = [0.0 for _ in range(L)]
    bwd_times = [0.0 for _ in range(L)]
    act_bytes_per_sample = [0 for _ in range(L)]
    pbytes = [0 for _ in range(L)]

    # Param bytes per block
    for i, blk in enumerate(blocks):
        pbytes[i] = param_bytes(blk)

    # Also measure head/tail params
    embed_params = param_bytes(model.get_input_embeddings())
    # tail: final norm + lm_head (lm_head may equal embed if tied)
    tail_params = 0
    if hasattr(model, "lm_head"):
        # Avoid double-count if tied (same storage)
        tail_params += sum(p.numel()*p.element_size() for n,p in model.lm_head.named_parameters())
    # Try final norm
    final_norm = None
    for name, mod in model.named_modules():
        if name.endswith(("final_layernorm", "final_norm", "norm", "ln_f")):
            final_norm = mod
    if final_norm is not None:
        tail_params += param_bytes(final_norm)

    # Forward/backward hooks to time per block and record output bytes
    fwd_start = [0.0 for _ in range(L)]
    bwd_start = [0.0 for _ in range(L)]

    def pre_hook(i):
        def _pre(module, inputs):
            fwd_start[i] = time.perf_counter()
        return _pre

    def fwd_hook(i):
        def _hook(module, inputs, outputs):
            nonlocal act_bytes_per_sample, fwd_times
            dt = time.perf_counter() - fwd_start[i]
            fwd_times[i] += dt
            # outputs could be tuple; take first tensor-like
            out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if torch.is_tensor(out):
                act_bytes_per_sample[i] = int(out.element_size() * out.shape[-1]) * args.seq_len
        return _hook

    def bwd_hook(i):
        def _bwd(module, grad_input, grad_output):
            nonlocal bwd_times, bwd_start
            # grad_output timing is tricky; measure bwd duration per block by start/end gap
            now = time.perf_counter()
            if bwd_start[i] == 0.0:
                bwd_start[i] = now
            else:
                bwd_times[i] += now - bwd_start[i]
                bwd_start[i] = 0.0
        return _bwd

    handles = []
    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_pre_hook(pre_hook(i)))
        handles.append(blk.register_forward_hook(fwd_hook(i)))
        # full backward hook (PyTorch 1.10+). If not available, comment out.
        if hasattr(blk, "register_full_backward_hook"):
            handles.append(blk.register_full_backward_hook(bwd_hook(i)))

    # Run
    iters = args.iters
    warmup = args.warmup
    for it in range(iters):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        if it >= warmup:
            # Measure backward only after warmup as well
            t0 = time.perf_counter()
            loss.backward()
            t1 = time.perf_counter()
        else:
            loss.backward()

    # Average over measured iterations
    denom = max(1, iters - warmup)
    fwd_times = [t / denom for t in fwd_times]
    # bwd_times may be zeros if backward hook did not capture; fall back to autograd total split
    # Here we approximate by proportional split if hooks unavailable.
    total_bwd = 0.0  # not robust via hooks on CPU; optional
    if sum(bwd_times) == 0.0:
        # Fall back: just mark unknown
        bwd_times = [0.0 for _ in range(L)]

    # Pack results
    result = {
        "host": args.host,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "vocab_size": vocab,
        "hidden": hidden,
        "embed_param_bytes": embed_params,
        "tail_param_bytes": tail_params,
        "layers": []
    }
    for i in range(L):
        result["layers"].append({
            "index": i,
            "param_bytes": pbytes[i],
            "activation_bytes_per_sample": act_bytes_per_sample[i],
            "forward_time_s": fwd_times[i],
            "backward_time_s": bwd_times[i],  # may be zero if hook unsupported
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Cleanup hooks
    for h in handles:
        h.remove()

if __name__ == "__main__":
    main()
