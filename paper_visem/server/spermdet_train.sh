#!/usr/bin/env bash
# Patch the remaining numpy-2 removals and launch SpermDet training on OUR
# official split (single class), following their recipe (img 640, batch 16).
set -uo pipefail
R=/root/autodl-tmp/spermdet_job/SpermDet
PY=/root/miniconda3/bin/python
cd "$R" || exit 1

echo "=== numpy2 aliases removed in 2.0 -> restore behaviour"
sed -i 's/np\.trapz/np.trapezoid/g; s/np\.bool\b/bool/g; s/np\.object\b/object/g' utils/*.py models/*.py 2>/dev/null
grep -rn "np.trapz" utils/*.py | head -3 || echo "  no np.trapz left"

echo "=== launch (100 epochs, their recipe; --nosave keeps disk small)"
rm -rf runs/train/spermdet_visem_c0
setsid nohup $PY train.py \
  --img 640 --batch 16 --epochs 100 \
  --data data/VISEM_official_c0.yaml --cfg models/SpermDet.yaml --weights '' \
  --workers 8 --device 0 --nosave \
  --project runs/train --name spermdet_visem_c0 \
  > /root/spermdet_train.log 2>&1 < /dev/null &
echo "launched pid $!"
sleep 90
echo "=== first 90 s of the log"
tr '\r' '\n' < /root/spermdet_train.log | grep -v "^ *$" | tail -12
echo "=== disk"
df -h /root/autodl-tmp | tail -1
