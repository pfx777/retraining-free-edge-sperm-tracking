#!/usr/bin/env bash
# Everything re-aligned to 50 epochs (same schedule for every model in the
# comparison table): our delivered recipe, the official YOLOv5n/x, and SpermDet.
# Two parallel streams (each fits in 24 GB at 448, batch 16).
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
R=/root/code
DATA=/root/code/datasets/visem_official
OFF=/root/autodl-tmp/visem_official_job/visem-tracking/sample_YOLO_models
SPD=/root/autodl-tmp/spermdet_job/SpermDet
CK=/root/autodl-tmp/shared_ckpt

echo "=== make sure the forks are patch-ready and the yamls exist"
for R2 in "$OFF" "$SPD"; do
  mkdir -p "$R2/ultralytics/yolo/utils"
  : > "$R2/ultralytics/__init__.py"; : > "$R2/ultralytics/yolo/__init__.py"
  : > "$R2/ultralytics/yolo/utils/__init__.py"
  printf 'def check_requirements(requirements=(), exclude=(), install=False):\n    return True\n\ndef check_version(current="0.0.0", minimum="0.0.0", **kwargs):\n    return True\n' \
    > "$R2/ultralytics/yolo/utils/checks.py"
  sed -i 's/\.astype(np\.int)/.astype(int)/g; s/np\.int\b/int/g; s/np\.float\b/float/g; s/np\.trapz/np.trapezoid/g; s/np\.bool\b/bool/g' "$R2"/utils/*.py "$R2"/models/*.py 2>/dev/null
done
printf 'train: %s/train/images\nval: %s/val/images\nnc: 1\nnames: [%s]\n' "$DATA" "$DATA" "'sperm'" > "$OFF/data/VISEM_official_c0.yaml"
printf 'train: %s/train/images\nval: %s/val/images\nnc: 1\nnames: [%s]\n' "$DATA" "$DATA" "'sperm'" > "$SPD/data/VISEM_official_c0.yaml"

cat > /root/streamA.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
SPD=/root/autodl-tmp/spermdet_job/SpermDet
echo "=== [A1] ours GatedCtx 448 / 50 epoch  $(date)"
bash /root/code/run_arch_arm.sh v8n_gctx448_ep50 /root/code/stock_yolov8n_gctx.yaml E26ep50 42 "EPOCHS=50"
echo "=== [A1] done rc=$? $(date)"
echo "=== [A2] SpermDet 448 / 50 epoch  $(date)"
cd "$SPD" && /root/miniconda3/bin/python train.py --img 448 --batch 16 --epochs 50 \
  --data data/VISEM_official_c0.yaml --cfg models/SpermDet.yaml --weights '' \
  --workers 8 --device 0 --nosave --project runs/train --name spermdet_448_ep50 --exist-ok 2>&1 | tr '\r' '\n' | tail -3
echo "=== [A2] done $(date)"
EOS

cat > /root/streamB.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
OFF=/root/autodl-tmp/visem_official_job/visem-tracking/sample_YOLO_models
PY=/root/miniconda3/bin/python
CK=/root/autodl-tmp/shared_ckpt
for W in yolov5n yolov5x; do
  [ -s "$CK/$W.pt" ] || curl -sL --max-time 900 -o "$CK/$W.pt" \
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/$W.pt"
done
cd "$OFF"
echo "=== [B1] yolov5n 448 / 50 epoch  $(date)"
$PY train.py --img 448 --batch 16 --epochs 50 --data data/VISEM_official_c0.yaml \
  --cfg models/yolov5n.yaml --weights "$CK/yolov5n.pt" --workers 8 --device 0 --nosave \
  --project runs/train --name y5n_448_ep50 --exist-ok 2>&1 | tr '\r' '\n' | tail -3
echo "=== [B1] done $(date)"
echo "=== [B2] yolov5x 448 / 50 epoch  $(date)"
$PY train.py --img 448 --batch 16 --epochs 50 --data data/VISEM_official_c0.yaml \
  --cfg models/yolov5x.yaml --weights "$CK/yolov5x.pt" --workers 8 --device 0 --nosave \
  --project runs/train --name y5x_448_ep50 --exist-ok 2>&1 | tr '\r' '\n' | tail -3
echo "=== [B2] done $(date)"
EOS
chmod +x /root/streamA.sh /root/streamB.sh

setsid nohup bash /root/streamA.sh > /root/streamA.log 2>&1 < /dev/null &
echo "streamA pid $!"
sleep 5
setsid nohup bash /root/streamB.sh > /root/streamB.log 2>&1 < /dev/null &
echo "streamB pid $!"
sleep 90
echo "=== streamA head"; tr '\r' '\n' < /root/streamA.log | grep -v "^ *$" | tail -4
echo "=== streamB head"; tr '\r' '\n' < /root/streamB.log | grep -v "^ *$" | tail -4
echo "=== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== disk"; df -h /root/autodl-tmp | tail -1
