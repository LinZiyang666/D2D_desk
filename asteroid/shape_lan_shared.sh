#!/usr/bin/env bash
# shape_lan_shared.sh
#
# Description:
#   Limit the total shared bandwidth for all LAN interfaces on a Linux router.
#   This script redirects ingress traffic from all LAN interfaces to a single
#   IFB device (ifb0) and applies traffic shaping there. The rate limit is
#   specified as a command-line argument in Mbps.
#
# Usage:
#   sudo bash shape_lan_shared.sh <rate_in_mbps>
# Example:
#   sudo bash shape_lan_shared.sh 400
#   This will limit the total ingress bandwidth of all LAN ports to 400 Mbps.

set -euo pipefail

# ===========================
# Parameter Validation
# ===========================
if [[ $# -ne 1 ]]; then
  echo "Usage: sudo bash $0 <rate_in_mbps>"
  echo "Example: sudo bash $0 400"
  exit 1
fi

RATE_MBPS=$1
# Ensure the input is a valid number
if ! [[ $RATE_MBPS =~ ^[0-9]+$ ]]; then
  echo "Error: The rate must be an integer (Mbps)."
  exit 1
fi

SHARED_RATE="${RATE_MBPS}mbit"  # Convert to tc-recognizable unit

# ===========================
# LAN Interfaces
# ===========================
LAN_IFS=(
  enp6s0f1   # 10.10.0.1/24
  enp4s0f0   # 10.11.0.1/24
  enp6s0f0   # 10.12.0.1/24
  enp4s0f1   # 10.13.0.1/24
  enp6s0f2   # 10.14.0.1/24
  enp6s0f3   # 10.15.0.1/24
)
IFB_DEV="ifb0"

# ===========================
# Root Privilege Check
# ===========================
if [[ $EUID -ne 0 ]]; then
  echo "Please run as root."
  exit 1
fi

echo "Applying total shared bandwidth limit: $SHARED_RATE"

# ===========================
# Load IFB Module
# ===========================
modprobe ifb numifbs=1 || true
ip link show "$IFB_DEV" &>/dev/null || ip link add "$IFB_DEV" type ifb
ip link set dev "$IFB_DEV" up

# ===========================
# Clear Existing Rules
# ===========================
tc qdisc del dev "$IFB_DEV" root 2>/dev/null || true
for dev in "${LAN_IFS[@]}"; do
  tc qdisc del dev "$dev" ingress 2>/dev/null || true
done

# ===========================
# Redirect LAN Ingress to IFB
# ===========================
for dev in "${LAN_IFS[@]}"; do
  echo "Configuring ingress redirection for $dev -> $IFB_DEV"
  tc qdisc add dev "$dev" handle ffff: ingress
  tc filter add dev "$dev" parent ffff: protocol all u32 \
    match u32 0 0 \
    action mirred egress redirect dev "$IFB_DEV"
done

# ===========================
# Apply Shaping on IFB
# ===========================
tc qdisc add dev "$IFB_DEV" root handle 1: htb default 10
tc class add dev "$IFB_DEV" parent 1: classid 1:1 htb rate "$SHARED_RATE" ceil "$SHARED_RATE"
tc class add dev "$IFB_DEV" parent 1:1 classid 1:10 htb rate "$SHARED_RATE" ceil "$SHARED_RATE"
tc qdisc add dev "$IFB_DEV" parent 1:10 handle 110: fq_codel

echo "Bandwidth limit applied successfully on all LAN interfaces: $SHARED_RATE"
