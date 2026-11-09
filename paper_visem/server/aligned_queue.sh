#!/usr/bin/env bash
# Train the third-party architectures on OUR official split with OUR schedule:
# img 448, batch 16, 30 epochs (the recipe behind the delivered model), so the
# comparison table is budget-aligned.  Queue (sequential, one GPU):
#   1) official YOLOv5n   (their code, COCO-pretrained weights)
#   2) SpermDet           (their code; no public pretrained weights)
#   3) official YOLOv5x   (their code, COCO-pretrained weights)
set -uo pipefail
OFF=/root/autodl-tmp/visem_official_job/visem-tracking/sample_YOLO_models
SPD=/root/autodl-tmp/spermdet_job/SpermDet
PY=/root/miniconda3/bin/python
DATA=/root/code/datasets/visem_official

echo "=== stop the 50-epoch SpermDet run"
pkill -f "train.py --img 640 --batch 16 --epochs 50" && echo "  signalled" || echo "  nothing to kill"
sleep 8
cd "$SPD" && [ -d runs/train/spermdet_visem_c0_ep50 ] && \
  mv runs/train/spermdet_visem_c0_ep50 "runs/train/spermdet_visem_c0_aborted50_$(date +%H%M)"

echo "=== make both forks runnable (legacy ultralytics import + numpy2 aliases)"
for R in "$OFF" "$SPD"; do
  mkdir -p "$R/ultralytics/yolo/utils"
  : > "$R/ultralytics/__init__.py"; : > "$R/ultralytics/yolo/__init__.py"
  : > "$R/ultralytics/yolo/utils/__init__.py"
  cat > "$R/ultralytics/yolo/utils/checks.py" <<'PY'
def check_requirements(requirements=(), exclude=(), install=False):
    return True
def check_version(current="0.0.0", minimum="0.0.0", **kwargs):
    return True
PY
  sed -i 's/\.astype(np\.int)/.astype(int)/g; s/np\.int\b/int/g; s/np\.float\b/float/g; s/np\.trapz/np.trapezoid/g; s/np\.bool\b/bool/g' \
    "$R"/utils/*.py "$R"/models/*.py 2>/dev/null
done
echo "  patched"

echo "=== data yaml for the official fork (our split, single class)"
cat > "$OFF/data/VISEM_official_c0.yaml" <<YML
train: $DATA/train/images
val: $DATA/val/images
nc: 1
names: ['sperm']
YML
cat "$OFF/data/VISEM_official_c0.yaml"

echo "=== COCO-pretrained initial weights (GitHub reachable)"
mkdir -p /root/autodl-tmp/shared_ckpt
for W in yolov5n.pt yolov5x.pt; do
  if [ ! -s "/root/autodl-tmp/shared_ckpt/$W" ]; then
    curl -sL --max-time 600 -o "/root/autodl-tmp/shared_ckpt/$W" \
      "https://github.com/ultralytics/yolov5/releases/download/v7.0/$W" \
      && echo "  $W $(du -h /root/autodl-tmp/shared_ckpt/$W | cut -f1)"
  else
    echo "  $W already present"
  fi
done

cat > /root/aligned_train.sh <<'EOS'
#!/usr/bin/env bash
set -uo pipefail
PY=/root/miniconda3/bin/python
OFF=/root/autodl-tmp/visem_official_job/visem-tracking/sample_YOLO_models
SPD=/root/autodl-tmp/spermdet_job/SpermDet
CK=/root/autodl-tmp/shared_ckpt
run() { echo "=== $1 $(date +%H:%M:%S)"; shift; "$@" 2>&1 | tr '\r' '\n' | grep -v "^ *$" | tail -4; }

cd "$OFF"
run "yolov5n 448/30ep" $PY train.py --img 448 --batch 16 --epochs 30 \
  --data data/VISEM_official_c0.yaml --cfg models/yolov5n.yaml --weights "$CK/yolov5n.pt" \
  --workers 8 --device 0 --nosave --project runs/train --name y5n_448_ep30 --exist-ok

cd "$SPD"
run "SpermDet 448/30ep" $PY train.py --img 448 --batch 16 --epochs 30 \
  --data data/VISEM_official_c0.yaml --cfg models/SpermDet.yaml --weights '' \
  --workers 8 --device 0 --nosave --project runs/train --name spermdet_448_ep30 --exist-ok

cd "$OFF"
run "yolov5x 448/30ep" $PY train.py --img 448 --batch 16 --epochs 30 \
  --data data/VISEM_official_c0.yaml --cfg models/yolov5x.yaml --weights "$CK/yolov5x.pt" \
  --workers 8 --device 0 --nosave --project runs/train --name y5x_448_ep30 --exist-ok

echo "=== ALL DONE $(date +%H:%M:%S)"
EOS
chmod +x /root/aligned_train.sh

setsid nohup bash /root/aligned_train.sh > /root/aligned_train.log 2>&1 < /dev/null &
echo "queued pid $!"
sleep 60
echo "=== first 60 s"
tr '\r' '\n' < /root/aligned_train.log | grep -v "^ *$" | tail -6
echo "=== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== disk"; df -h /root/autodl-tmp | tail -1
