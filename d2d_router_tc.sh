#!/usr/bin/env bash
# D2D pairwise shaping on router egress for 6 nodes (units: Mbps)
# node-i: 10.(10+i).0.2/24  <->  router-if: 10.(10+i).0.1/24  for i=0..5
# Actions: apply | show | clear

set -euo pipefail

# --------- CLI parsing ---------
usage() {
  cat <<'USAGE'
Usage:
  sudo ./d2d_router_tc.sh apply [--default N] [--override S D N ...] [--o S:D=N ...] [--from-file FILE]
  sudo ./d2d_router_tc.sh show
  sudo ./d2d_router_tc.sh clear

Options:
  --default N          Default per-pair rate in Mbps for all src->dst (except self). Default: 20
  --override S D N     Override one directed pair src=S, dst=D to N Mbps; can repeat.
  --o S:D=N            Shorthand for --override (can repeat), e.g. --o 1:3=25
  --from-file FILE     Load overrides from FILE; lines like: "S D N" or "S,D,N".
                       Lines starting with # are ignored. Indices S,D in [0..5].
Examples:
  apply with defaults only (20 Mbps):
    sudo ./d2d_router_tc.sh apply
  apply with default 15 Mbps and three overrides:
    sudo ./d2d_router_tc.sh apply --default 15 --override 0 1 50 --override 2 5 8 --o 3:4=12
  inspect current shaping:
    sudo ./d2d_router_tc.sh show
  clear all shaping:
    sudo ./d2d_router_tc.sh clear
USAGE
}

ACTION="${1:-}"
[[ -z "$ACTION" || "$ACTION" == "-h" || "$ACTION" == "--help" ]] && { usage; exit 0; }
shift || true

DEFAULT_RATE_MBPS=20
declare -a OV_S=()
declare -a OV_D=()
declare -a OV_R=()

add_override() { OV_S+=("$1"); OV_D+=("$2"); OV_R+=("$3"); }

parse_file() {
  local f="$1"; [[ -f "$f" ]] || { echo "ERROR: no such file: $f" >&2; exit 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    [[ -z "$line" ]] && continue
    # accept "S D N" or "S,D,N"
    line="${line//,/ }"
    read -r s d n <<<"$line"
    [[ "$s" =~ ^[0-5]$ && "$d" =~ ^[0-5]$ && "$s" != "$d" ]] || { echo "bad line: $line" >&2; exit 1; }
    [[ "$n" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "bad Mbps: $n" >&2; exit 1; }
    add_override "$s" "$d" "$n"
  done < "$f"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --default) shift; DEFAULT_RATE_MBPS="${1:-}"; [[ "$DEFAULT_RATE_MBPS" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "bad --default" >&2; exit 1; } ;;
    --override) shift; s="${1:-}"; d="${2:-}"; n="${3:-}"; [[ "$s" =~ ^[0-5]$ && "$d" =~ ^[0-5]$ && "$s" != "$d" ]] || { echo "bad --override S D" >&2; exit 1; }; [[ "$n" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "bad Mbps" >&2; exit 1; }; add_override "$s" "$d" "$n"; shift 2 ;;
    --o) shift; trip="${1:-}"; [[ "$trip" =~ ^([0-5]):([0-5])=([0-9]+([.][0-9]+)?)$ ]] || { echo "bad --o S:D=N" >&2; exit 1; }; s="${BASH_REMATCH[1]}"; d="${BASH_REMATCH[2]}"; n="${BASH_REMATCH[3]}"; [[ "$s" != "$d" ]] || { echo "src!=dst" >&2; exit 1; }; add_override "$s" "$d" "$n" ;;
    --from-file) shift; parse_file "${1:-}";;
    --help|-h) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
  shift || true
done

[[ "$ACTION" =~ ^(apply|show|clear)$ ]] || { echo "Invalid ACTION: $ACTION" >&2; usage; exit 1; }

# --------- helpers ---------
need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing dependency: $1" >&2; exit 1; }; }
need ip; need tc; need awk

router_ip_of() { local d="$1"; echo "10.$((10+d)).0.1"; }
node_ip_of()   { local i="$1"; echo "10.$((10+i)).0.2"; }

dev_for_router_ip() {
  local ip="$1"
  ip -o -4 addr show | awk -v pat=" ${ip}/" '$0 ~ pat {print $2; found=1} END{exit found?0:1}'
}

declare -A IFACE
resolve_ifaces() {
  for d in {0..5}; do
    local rip dev
    rip=$(router_ip_of "$d")
    if ! dev=$(dev_for_router_ip "$rip"); then
      echo "ERROR: cannot find device that has ${rip}/24" >&2
      exit 1
    fi
    IFACE["$d"]="$dev"
  done
}

mbps_to_tc() { local v="$1"; printf "%smbit" "$v"; }

get_rate_mbps() {
  local s="$1" d="$2"
  # search overrides
  for i in "${!OV_S[@]}"; do
    [[ "${OV_S[$i]}" == "$s" && "${OV_D[$i]}" == "$d" ]] && { echo "${OV_R[$i]}"; return; }
  done
  echo "$DEFAULT_RATE_MBPS"
}

install_for_dst() {
  local d="$1" dev="${IFACE[$d]}"
  tc qdisc del dev "$dev" root 2>/dev/null || true
  tc qdisc add dev "$dev" root handle 1: htb default 999
  tc class add dev "$dev" parent 1: classid 1:1 htb rate 10gbit ceil 10gbit
  for s in {0..5}; do
    [[ "$s" == "$d" ]] && continue
    local rate_mbps rate_tc minor classid src dst
    rate_mbps=$(get_rate_mbps "$s" "$d")
    rate_tc=$(mbps_to_tc "$rate_mbps")
    minor=$((d*10 + s + 1))
    classid="1:${minor}"
    tc class add dev "$dev" parent 1:1 classid "$classid" htb rate "$rate_tc" ceil "$rate_tc"
    tc qdisc add dev "$dev" parent "$classid" handle "${minor}:" fq_codel
    src=$(node_ip_of "$s"); dst=$(node_ip_of "$d")
    tc filter add dev "$dev" protocol ip parent 1: prio 1 u32 \
      match ip src "${src}/32" match ip dst "${dst}/32" flowid "$classid"
  done
  tc class add dev "$dev" parent 1: classid 1:999 htb rate 1kbit ceil 1kbit 2>/dev/null || true
}

show_status() {
  for d in {0..5}; do
    local dev="${IFACE[$d]}"
    echo "==== ${dev} -> node-${d} $(node_ip_of "$d") ===="
    tc -s class show dev "$dev" || true
    tc filter show dev "$dev" parent 1: || true
    echo
  done
}

clear_all() {
  for d in {0..5}; do
    local dev="${IFACE[$d]}"
    tc qdisc del dev "$dev" root 2>/dev/null || true
  done
}

print_plan() {
  echo "Default per-pair rate: ${DEFAULT_RATE_MBPS} Mbps"
  if [[ "${#OV_S[@]}" -gt 0 ]]; then
    echo "Overrides:"
    for i in "${!OV_S[@]}"; do
      echo "  ${OV_S[$i]} -> ${OV_D[$i]} = ${OV_R[$i]} Mbps"
    done
  else
    echo "Overrides: (none)"
  fi
  echo
  echo "Interface mapping:"
  for d in {0..5}; do
    echo "  dst ${d}: dev=${IFACE[$d]}  router_ip=$(router_ip_of "$d")  node_ip=$(node_ip_of "$d")"
  done
  echo
}

# --------- main ---------
resolve_ifaces

case "$ACTION" in
  apply)
    print_plan
    for d in {0..5}; do install_for_dst "$d"; done
    echo "Applied pairwise shaping on all 6 egress interfaces (Mbps)."
    ;;
  show)
    print_plan
    show_status
    ;;
  clear)
    clear_all
    echo "Cleared all qdisc roots on the 6 interfaces."
    ;;
esac
