#!/usr/bin/env bash
# Same comparison, but with the DELIVERED budget: 448 / batch 16 / 30 epochs,
# i.e. exactly the recipe behind our shipped model.  Reuses the streamA/streamB
# log paths and the [xx] done markers so the existing watcher keeps working.
#   stream A: SpermDet 448/30ep
#   stream B: official YOLOv5n 448/30ep  ->  official YOLOv5x 448/30ep
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
OFF=/root/autodl-tmp/visem_official_job/visem-tracking/sample_YOLO_models
SPD=/root/autodl-tmp/spermdet_job/SpermDet
CK=/root/autodl-tmp/shared_ckpt
PY=/root/miniconda3/bin/python

cat > /root/streamA.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
SPD=/root/autodl-tmp/spermdet_job/SpermDet
echo "=== [A2] SpermDet 448 / 30 epoch  $(date)"
cd "$SPD" && /root/miniconda3/bin/python train.py --img 448 --batch 16 --epochs 30 \
  --data data/VISEM_official_c0.yaml --cfg models/SpermDet.yaml --weights '' \
  --workers 8 --device 0 --nosave --project runs/train --name spermdet_448_ep30 --exist-ok 2>&1 | tr '\r' '\n' | tail -3
echo "=== [A2] done rc=$? $(date)"
EOS

cat > /root/streamB.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
OFF=/root/autodl-tmp/visem_official_job/visem-tracking/sample_YOLO_models
CK=/root/autodl-tmp/shared_ckpt
PY=/root/miniconda3/bin/python
cd "$OFF"
echo "=== [B1] yolov5n 448 / 30 epoch  $(date)"
$PY train.py --img 448 --batch 16 --epochs 30 --data data/VISEM_official_c0.yaml \
  --cfg models/yolov5n.yaml --weights "$CK/yolov5n.pt" --workers 8 --device 0 --nosave \
  --project runs/train --name y5n_448_ep30 --exist-ok 2>&1 | tr '\r' '\n' | tail -3
echo "=== [B1] done $(date)"
echo "=== [B2] yolov5x 448 / 30 epoch  $(date)"
$PY train.py --img 448 --batch 16 --epochs 30 --data data/VISEM_official_c0.yaml \
  --cfg models/yolov5x.yaml --weights "$CK/yolov5x.pt" --workers 8 --device 0 --nosave \
  --project runs/train --name y5x_448_ep30 --exist-ok 2>&1 | tr '\r' '\n' | tail -3
echo "=== [B2] done $(date)"
EOS
chmod +x /root/streamA.sh /root/streamB.sh

pkill -f "train.py --img 448" 2>/dev/null || true
sleep 3
setsid nohup bash /root/streamA.sh > /root/streamA.log 2>&1 < /dev/null &
echo "streamA pid $!"
sleep 5
setsid nohup bash /root/streamB.sh > /root/streamB.log 2>&1 < /dev/null &
echo "streamB pid $!"
sleep 75
echo "=== A"; tr '\r' '\n' < /root/streamA.log | grep -v '^ *$' | tail -3
echo "=== B"; tr '\r' '\n' < /root/streamB.log | grep -v '^ *$' | tail -3
echo "=== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== disk"; df -h /root/autodl-tmp | tail -1
