#!/usr/bin/env bash
# Restart SpermDet training with a 50-epoch budget (aligned with the official
# YOLOv5 baseline's 50 epochs).  The aborted 100-epoch run is kept aside.
set -uo pipefail
R=/root/autodl-tmp/spermdet_job/SpermDet
PY=/root/miniconda3/bin/python
cd "$R" || exit 1

echo "=== stop the 100-epoch run"
pkill -f "train.py --img 640 --batch 16 --epochs 100" && echo "  signalled" || echo "  nothing to kill"
sleep 8
pgrep -af "train.py" | grep -v pgrep || echo "  no train.py running"

echo "=== keep the aborted run aside (no deletion)"
if [ -d runs/train/spermdet_visem_c0 ]; then
  mv runs/train/spermdet_visem_c0 "runs/train/spermdet_visem_c0_aborted100_$(date +%H%M)"
fi
ls runs/train

echo "=== relaunch: their recipe, 50 epochs (schedule aligned to the official baseline)"
setsid nohup $PY train.py \
  --img 640 --batch 16 --epochs 50 \
  --data data/VISEM_official_c0.yaml --cfg models/SpermDet.yaml --weights '' \
  --workers 8 --device 0 --nosave \
  --project runs/train --name spermdet_visem_c0_ep50 \
  > /root/spermdet_train50.log 2>&1 < /dev/null &
echo "launched pid $!"
sleep 75
echo "=== first 75 s"
tr '\r' '\n' < /root/spermdet_train50.log | grep -v "^ *$" | tail -6
echo "=== gpu / disk"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
df -h /root/autodl-tmp | tail -1
