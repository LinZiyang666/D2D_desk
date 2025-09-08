#!/bin/bash
# -------------- 带宽复原 -----------------
NIC="${PP_BW_IF:-eth0}"                     # 同 Python 端的 NIC 变量
ORIG_FILE="/tmp/pp_orig_qdisc_${NIC}.txt"

echo ">>> Restoring bandwidth on $NIC …"

# 1) 若根 qdisc 是 tbf（训练脚本设置），先删掉即可解除限速
if tc qdisc show dev "$NIC" | grep -q '\<tbf\>'; then
  echo "Found tbf qdisc on $NIC, deleting it"
  sudo tc qdisc del dev "$NIC" root 2>/dev/null
fi

# 2) 如果有备份文件，再按备份逐行恢复
if [[ -f "$ORIG_FILE" ]]; then
  echo "Applying backed‑up qdisc configuration from $ORIG_FILE"
  # 备份文件是一行行的 'tc qdisc add …' 或 'tc qdisc replace …'
  # 为安全起见先清空根 qdisc，再重放
  sudo tc qdisc del dev "$NIC" root 2>/dev/null
  while IFS= read -r line; do
    # 忽略空行 / 注释
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    sudo $line 2>/dev/null
  done < "$ORIG_FILE"
  rm -f "$ORIG_FILE"
  echo "Restored and removed backup file."
else
  echo "No backup qdisc file found, root qdisc now uses kernel default."
fi
echo ">>> Bandwidth restoration finished."
# -------------- 带宽复原 END --------------

PORT=29500

# Search for and kill the processes occupying the specified port
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
  echo "Killing process on port $PORT with PID: $PID"
  kill -9 $PID
else
  echo "No process is using port $PORT"
fi

echo "Killing all torchrun processes..."
pkill -9 -f torchrun || echo "No torchrun process found."

echo "Killing all python processes..."
pkill -9 -f python || echo "No python process found."

echo "Killing all gloo-related processes..."
pgrep -f gloo | xargs -r kill -9 || echo "No gloo process found."

echo "Killing all nccl-related processes..."
pgrep -f nccl | xargs -r kill -9 || echo "No nccl process found."

echo "Done."

