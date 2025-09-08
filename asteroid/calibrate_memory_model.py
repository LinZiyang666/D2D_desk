#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit a stage peak-memory model from measured RSS on sampled spans.

Model:
  peak_bytes ≈ base + alpha * sum(weights_bytes) + Kp * sum(activation_boundary_bytes_per_sample) * batch

We solve [base, alpha, Kp] via least squares using S sampled spans.

Usage:
  python calibrate_memory_model.py \
      --layers_json layers_cpu1.json \
      --model_path /path/to/qwen-0_6b \
      --samples 20 \
      --out memory_model.json

Notes:
- Runs on a SINGLE machine; the fitted [base, alpha, Kp] are model/training-config specific.
- It executes only the selected span blocks forward+backward to record RSS delta.
"""

import argparse, json, random, time
from typing import List, Tuple

import torch
import torch.nn as nn
import psutil
import numpy as np

def find_blocks(model: nn.Module) -> List[nn.Module]:
    for attr in ["model", "transformer", "gpt_neox", "backbone"]:
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if hasattr(inner, "layers"):
                layers = getattr(inner, "layers")
                if isinstance(layers, (list, nn.ModuleList)):
                    return list(layers)
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, (list, nn.ModuleList)):
            return list(mod)
    raise RuntimeError("Cannot find blocks")

def run_span(blocks: List[nn.Module], s: int, e: int, B: int, T: int, vocab: int) -> int:
    # Create a tiny subgraph: input hidden state -> blocks[s:e] -> simple LM head-ish loss
    # We use random embeddings to avoid needing the full model forward.
    hidden = next(blocks[0].parameters()).shape[-1]
    x = torch.randn(B, T, hidden, dtype=torch.float32)
    x.requires_grad_(True)
    y = x
    for i in range(s, e):
        y = blocks[i](y)[0] if isinstance(blocks[i](y), (tuple, list)) else blocks[i](y)
    # simple scalar loss
    loss = y.sum()
    loss.backward()
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers_json", required=True, help="output from measure_layers.py")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.layers_json, "r"))
    B = data["batch"]; T = data["seq_len"]
    layers = data["layers"]
    L = len(layers)

    # Load model to get actual blocks
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=False)
    model.train()

    blocks = find_blocks(model)

    # Prepare regression matrices
    S = max(5, min(args.samples, L))
    spans = []
    for _ in range(S):
        s = random.randint(0, L-1)
        e = random.randint(s+1, L)
        spans.append((s,e))

    X = []
    y = []
    proc = psutil.Process()

    for (s,e) in spans:
        sum_w = sum(layers[i]["param_bytes"] for i in range(s,e))
        sum_a = sum(layers[i]["activation_bytes_per_sample"] for i in range(s,e))

        rss0 = proc.memory_info().rss
        run_span(blocks, s, e, B, T, data["vocab_size"])
        time.sleep(0.05)
        rss1 = proc.memory_info().rss
        peak = max(rss1 - rss0, 0)  # rough delta; not perfect but indicative

        X.append([1.0, float(sum_w), float(sum_a * B)])
        y.append(float(peak))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Solve least squares: [base, alpha, Kp]
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    base, alpha, Kp = theta.tolist()
    json.dump({"base": base, "alpha": alpha, "Kp": Kp, "batch": B, "seq_len": T}, open(args.out,"w"), indent=2)

if __name__ == "__main__":
    main()
