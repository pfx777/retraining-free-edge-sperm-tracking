#!/usr/bin/env bash
# Dump official-checkpoint detections: val for both checkpoints (comparison) and
# train for both (needed to fit the device constant kappa for THEIR model).
set -uo pipefail
JOB=/root/autodl-tmp/visem_official_job
DATA=/root/code/datasets/visem_official
PY=/root/miniconda3/bin/python
cd "$JOB" || exit 1

run() {  # run <tag> <weights> <src> <out>
  echo "=== $1  $(date +%H:%M:%S)"
  $PY /root/infer_dump.py --weights "ckpt/$2" --src "$3" --out "$4" 2>&1 | tail -8
}

run "val yolov5n"  yolov5n_best.pt "$DATA/val/images"   dets_val_yolov5n
run "train yolov5n" yolov5n_best.pt "$DATA/train/images" dets_train_yolov5n
run "train yolov5x" yolov5x_best.pt "$DATA/train/images" dets_train_yolov5x

echo "=== packaging $(date +%H:%M:%S)"
tar czf official_dets.tgz dets_val_yolov5x dets_val_yolov5n dets_train_yolov5x dets_train_yolov5n
ls -la official_dets.tgz
du -sh dets_*
echo "=== done $(date +%H:%M:%S)"
