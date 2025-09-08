"""
Run HPP planner for Qwen3-0.6B using the user's Algorithm 1 & 2 implementations.
This version explicitly accounts for:
  - embed_tokens parameters folded into the FIRST block
  - final_norm (+ lm_head) parameters folded into the LAST block
  - last block's boundary activation set to 0 (no outgoing activation)

Place this file alongside:
  - asteroid_Batch_plan.py
  - asteroid_Hpp_plan.py

Outputs will be written to ./out/: stages.csv, steps.csv, plan_meta.json, summary.txt
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# --- Import the user's implementations ---
from asteroid_Hpp_plan import Device, plan_hpp_dynamic
from asteroid_Batch_plan import allocate_microbatch_samples  # Algorithm 1


# =====================================
# ============== CONFIG ===============
# =====================================

# Core model shape (approximate for byte sizing)
L = 28                # number of transformer blocks (excludes embed/tail, handled below)
HIDDEN = 1024         # hidden size used for sizing estimates
FF_MULT = 2.6667      # SwiGLU effective expansion factor
DTYPE_BYTES = 2       # 2 for bf16/FP16, 4 for FP32

# Sequence length you plan to train on (affects activation bytes)
SEQ_LEN = 2048

# Micro-batch planning
MICRO_BATCH_SIZE = 16     # B
NUM_MICRO_BATCHES = 8     # M (1F1B rounds)
MAX_STAGES = None         # or set to <= min(L, N)

# Device inventory (8 CPU machines). Fill in your real numbers.
memory_budget_gb = [24, 24, 24, 24, 24, 24, 24, 24]  # per-machine peak RAM allowed to the training process
capacity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # relative compute speed (higher = faster)

# Network: uniform full-mesh link bandwidth in Gbps
LINK_GBPS = 10.0

# === NEW: Head/Tail components ===
# Replace VOCAB_SIZE with your tokenizer's actual size. If weights are tied, set TIED_EMBED=True.
VOCAB_SIZE = 150_000      # <-- REPLACE with your actual vocab size (e.g., from tokenizer/config)
TIED_EMBED = True         # True if lm_head shares weights with embed_tokens (common in LMs)

# Output directory
OUT_DIR = Path("./out")


# =====================================
# ======= MODEL SIZE ESTIMATION =======
# =====================================
def estimate_qwen3_06b_per_layer_bytes(L: int, hidden: int, ff_mult: float, dtype_bytes: int) -> Tuple[List[int], List[int]]:
    """
    Rough bytes estimates per transformer block:
      - weights_size[l]: parameter bytes per layer
      - activation_size[l]: boundary activation bytes per sample (we multiply by seq_len later)
    """
    # Attention (Q,K,V,O): 4 * hidden * hidden
    attn_params = 4 * hidden * hidden
    # MLP (SwiGLU): two "up" + one "down"
    ff = int(ff_mult * hidden)
    mlp_params = (hidden * ff) * 2 + (ff * hidden)

    per_layer_params = attn_params + mlp_params
    weights_size = [per_layer_params * dtype_bytes for _ in range(L)]

    # Activation bytes per token across boundary (per sample). We'll scale by seq_len later.
    activation_unit = hidden * dtype_bytes
    activation_size = [activation_unit for _ in range(L)]
    return weights_size, activation_size


def build_activation_sizes(base_per_layer: List[int], seq_len: int) -> List[int]:
    return [b * seq_len for b in base_per_layer]


# =====================================
# ======= DEVICE & NETWORK SETUP ======
# =====================================
def build_devices(mem_gb: List[float], caps: List[float]) -> List[Device]:
    assert len(mem_gb) == len(caps), "memory_budget_gb and capacity must have same length"
    devices: List[Device] = []
    for i, (m, c) in enumerate(zip(mem_gb, caps)):
        devices.append(Device(name=f"cpu{i+1}", memory_budget=int(m * (1024**3)), capacity=float(c)))
    return devices


def build_bandwidth_matrix(devices: List[Device], link_gbps: float) -> Dict[Tuple[str, str], float]:
    bytes_per_sec = link_gbps * 1e9 / 8.0  # Gbps -> bytes/sec
    names = [d.name for d in devices]
    bw: Dict[Tuple[str, str], float] = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            bw[(a, b)] = bytes_per_sec
    return bw


# =====================================
# ============= LATENCY ===============
# =====================================
def make_latency_of_layer(dev_caps: Dict[str, float], L: int):
    """
    Simple latency model:
      - base forward/backward time for block l at batch=16 on a device with cap=1.0,
        then scale linearly with batch and inversely with capacity.
    Replace with measurements if you have them.
    """
    # base times (seconds) @ batch=16, cap=1.0
    base_fp = [0.010 + 0.0002 * (i % 8) for i in range(L)]
    base_bp = [0.020 + 0.0004 * (i % 8) for i in range(L)]

    def latency_of_layer(dev_name: str, layer_idx: int, batch: int):
        cap = max(dev_caps.get(dev_name, 1.0), 1e-6)
        tf = base_fp[layer_idx] * (batch / 16) / cap
        tb = base_bp[layer_idx] * (batch / 16) / cap
        return tf, tb

    return latency_of_layer


# =====================================
# ============== MEMORY ===============
# =====================================
def make_memory_of_stage(weights_size: List[int], activation_size: List[int]):
    """
    Peak memory model per stage on a device:
      base + 2 * (sum weights) + Kp * (sum activations) * batch
    """
    Kp = 2.0
    base_mem = 20_000_000  # bytes; graph/optimizer overhead per device per stage

    def memory_of_stage(dev_name: str, span: Tuple[int, int], batch: int) -> int:
        s, e = span
        sum_w = sum(weights_size[l] for l in range(s, e))
        sum_a = sum(activation_size[l] for l in range(s, e))
        peak = base_mem + 2 * sum_w + int(Kp * sum_a) * batch
        return peak

    return memory_of_stage


# =====================================
# =============== MAIN ================
# =====================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Transformer block sizes (bytes)
    weights_size, activation_base = estimate_qwen3_06b_per_layer_bytes(L, HIDDEN, FF_MULT, DTYPE_BYTES)
    activation_size = build_activation_sizes(activation_base, SEQ_LEN)

    # === Head/Tail accounting (Scheme A) ===============================
    # embed_tokens parameters -> first block
    EMBED_BYTES  = VOCAB_SIZE * HIDDEN * DTYPE_BYTES
    # RMSNorm scale vector is small; lm_head may be tied to embed
    NORM_BYTES   = HIDDEN * DTYPE_BYTES
    LMHEAD_BYTES = 0 if TIED_EMBED else (VOCAB_SIZE * HIDDEN * DTYPE_BYTES)

    weights_size[0]  += EMBED_BYTES
    weights_size[-1] += (NORM_BYTES + LMHEAD_BYTES)

    # The last block has no outgoing activation boundary
    activation_size[-1] = 0
    # ===================================================================

    # 2) Devices & network
    devices = build_devices(memory_budget_gb, capacity)
    dev_caps = {d.name: d.capacity for d in devices}
    bandwidth_matrix = build_bandwidth_matrix(devices, LINK_GBPS)

    # 3) Latency & Memory callbacks
    latency_of_layer = make_latency_of_layer(dev_caps, L)
    memory_of_stage = make_memory_of_stage(weights_size, activation_size)

    # 4) Run Algorithm 2 (which calls Algorithm 1 inside)
    plan = plan_hpp_dynamic(
        num_layers=L,
        weights_size=weights_size,
        activation_size=activation_size,
        devices=devices,
        bandwidth_matrix=bandwidth_matrix,
        micro_batch_size=MICRO_BATCH_SIZE,
        num_micro_batches=NUM_MICRO_BATCHES,
        latency_of_layer=latency_of_layer,
        memory_of_stage=memory_of_stage,
        allocate_fn=allocate_microbatch_samples,  # Algorithm 1
        block_size=1,
        max_stages=MAX_STAGES,
    )

    # 5) Save outputs
    # 5.1 stages.csv
    all_dev_names = [d.name for d in devices]
    stage_rows = []
    for i, st in enumerate(plan.stages):
        row = {
            "stage_id": i,
            "layers": f"[{st.layer_start}, {st.layer_end})",
            "devices": ",".join(st.device_names),
            "Ef(s)": round(st.Ef, 6),
            "Eb(s)": round(st.Eb, 6),
            "Ta(s)": round(st.Ta, 6),
        }
        for name in all_dev_names:
            row[f"y[{name}]"] = st.y_allocation.get(name, 0)
        stage_rows.append(row)
    df_stages = pd.DataFrame(stage_rows)
    df_stages.to_csv(OUT_DIR / "stages.csv", index=False)

    # 5.2 steps.csv
    Ef, Eb, Ta = plan.meta["Ef"], plan.meta["Eb"], plan.meta["Ta"]
    Tw, Te, totals = plan.meta["Tw"], plan.meta["Te"], plan.meta["totals"]
    dm = plan.meta["dominant_step_index"]

    steps = []
    for s in range(len(Ef)):
        steps.append({
            "step_id": s,
            "Ef": round(Ef[s], 6),
            "Eb": round(Eb[s], 6),
            "Ta": round(Ta[s], 6),
            "Tw": round(Tw[s], 6),
            "Te": round(Te[s], 6),
            "Tw+Te+Ta": round(totals[s], 6),
            "dominant": bool(s == dm),
        })
    df_steps = pd.DataFrame(steps)
    df_steps.to_csv(OUT_DIR / "steps.csv", index=False)

    # 5.3 plan_meta.json
    meta = {
        "round_latency_s": plan.round_latency,
        "dominant_step_index": plan.meta["dominant_step_index"],
        "Ef": Ef, "Eb": Eb, "Ta": Ta,
        "Tw": Tw, "Te": Te, "totals": totals,
    }
    with open(OUT_DIR / "plan_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 5.4 summary.txt
    lines = []
    lines.append(f"Chosen number of stages: {len(plan.stages)}")
    lines.append(f"Pipeline round latency (objective): {plan.round_latency:.6f} seconds")
    lines.append("Stages:")
    for i, st in enumerate(plan.stages):
        lines.append(f"  - Stage {i}: layers [{st.layer_start}, {st.layer_end}), Ef={st.Ef:.6f}, Eb={st.Eb:.6f}, Ta={st.Ta:.6f}")
        for name in st.device_names:
            lines.append(f"      y[{name}] = {st.y_allocation.get(name, 0)}")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWROTE: {OUT_DIR/'stages.csv'}")
    print(f"WROTE: {OUT_DIR/'steps.csv'}")
    print(f"WROTE: {OUT_DIR/'plan_meta.json'}")
    print(f"WROTE: {OUT_DIR/'summary.txt'}")


if __name__ == "__main__":
    main()