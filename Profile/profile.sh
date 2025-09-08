#!/usr/bin/env bash
set -euo pipefail

# Optional: check systemd-run exists
command -v systemd-run >/dev/null || { echo "systemd-run not found"; exit 1; }

# -------- Config --------
BATCH_SIZES=(1 2 3 4 5)
UTILS=(100 70)   # target % of the whole machine
# ------------------------

CORES=$(nproc --all)
PWD_NOW="$(pwd)"
echo "Detected CPU cores: ${CORES}"
echo "Working directory : ${PWD_NOW}"

for util in "${UTILS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    echo "=============================================="
    echo "Run: CPU util ${util}% of machine  |  batch ${bs}"
    echo "=============================================="

    # systemd CPUQuota is % of a single CPU.
    # To target X% of the entire machine, multiply by core count.
    QUOTA_PCT=$(( util * CORES ))%

    OUTFILE="./cpu${util}_bs${bs}.json"

    # --scope: transient scope cgroup
    # --wait/--collect: wait for completion & collect status
    # -p WorkingDirectory: run in current dir so outputs land here
    systemd-run --scope --quiet \
      -p "CPUQuota=${QUOTA_PCT}" \
      -- python measure_layers.py --batch "${bs}" --out "${OUTFILE}" --host "cpu${util}"
  done
done

echo "All runs completed."
