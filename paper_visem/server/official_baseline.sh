#!/usr/bin/env bash
# Server-side job: reproduce the official VISEM-Tracking baseline and dump its
# detections on the four validation videos, so the host can (a) compare
# protocols and (b) transfer the label-free calibration to a third-party detector.
#
# Run on the GPU server (AutoDL), from any directory:
#     bash official_baseline.sh 2>&1 | tee official_baseline.log
#
# Outputs under $WORK/visem_official/:
#   dets/official_<vid>.txt     frame cx cy w h score   (normalised, class-0)
#   eval/official_eval.log      official evaluation script output
#   official_dets.tgz           tarball of dets/ (send this back)
set -uo pipefail

WORK="${WORK:-/root/autodl-tmp/visem_official}"
REPO="$WORK/visem-official"
mkdir -p "$WORK"/{dets,eval,frames}
cd "$WORK" || exit 1

echo "== 0. environment"
date; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU visible"
df -h "$WORK" | tail -1
python3 -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || true

echo "== 1. official repository (MIT)"
if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 https://github.com/simulamet-host/visem-tracking "$REPO" || echo "clone failed"
fi
ls "$REPO" | head
cat "$REPO/sample_YOLO_models/VISEM_Tracking_after_index_corrections.yaml" 2>/dev/null | head -20

echo "== 2. official pre-trained checkpoints (HuggingFace dataset repo)"
python3 - <<'PY'
import json, urllib.request, os
api = "https://huggingface.co/api/datasets/SimulaMet-HOST/VISEM-Tracking/tree/main/best_checkpoints"
out = os.environ.get("WORK", "/root/autodl-tmp/visem_official") + "/ckpt"
os.makedirs(out, exist_ok=True)
try:
    with urllib.request.urlopen(api, timeout=60) as fh:
        items = json.load(fh)
except Exception as exc:
    print("  HF listing failed:", exc); items = []
for it in items:
    if it.get("type") != "file":
        continue
    name = it["path"].split("/")[-1]
    url = "https://huggingface.co/datasets/SimulaMet-HOST/VISEM-Tracking/resolve/main/best_checkpoints/" + name
    dst = os.path.join(out, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print("  have", name); continue
    try:
        urllib.request.urlretrieve(url, dst)
        print("  got ", name, os.path.getsize(dst))
    except Exception as exc:
        print("  FAIL", name, exc)
PY
ls -la "$WORK/ckpt" 2>/dev/null | head

echo "== 3. count what the official checkpoints are"
echo "    (their training used YOLOv5 n/s/m/l/x at img 640, 50 epochs, batch 16)"

echo "== 4. NOTE for the operator"
cat <<'EOF'
  The official repo expects its own dataset layout (see the yaml above) and its
  own YOLOv5 fork for detect.py.  Two supported routes:
    (a) if the VISEM frames in yolo layout are already on this machine, point
        VISEM_YOLO_ROOT at them and re-run with RUN_DETECT=1:
            VISEM_YOLO_ROOT=/path/to/to_yolo5_..._corrected_index RUN_DETECT=1 bash official_baseline.sh
    (b) otherwise send the host the checkpoint path and the 4 val videos and let
        the host run the dump with its own loader (same normalised MOT format).
  Either way the deliverable is dets/official_<vid>.txt in:
      frame cx cy w h score
  with coordinates normalised to the 640x480 frame, which is what the project's
  MOTA scorer and the calibration fitter consume.
EOF

if [ "${RUN_DETECT:-0}" = "1" ] && [ -n "${VISEM_YOLO_ROOT:-}" ]; then
  echo "== 5. inference with the official detect.py"
  cd "$REPO/sample_YOLO_models" || exit 1
  for V in 52 54 60 82; do
    for W in "$WORK"/ckpt/*.pt; do
      [ -e "$W" ] || continue
      TAG="$(basename "$W" .pt)"
      echo "--- $V with $TAG"
      python3 detect.py --img 640 --weights "$W" \
        --source "$VISEM_YOLO_ROOT/$V/images" \
        --hide-conf --hide-labels --save-txt --save-conf --device 0 \
        --project "$WORK/eval" --name "${TAG}_$V" 2>&1 | tail -5
    done
  done
  echo "== 6. convert label txt -> single MOT file per video"
  python3 - "$WORK" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
for exp in sorted((root / "eval").glob("*_*")):
    labels = exp / "labels"
    if not labels.is_dir():
        continue
    vid = exp.name.split("_")[-1]
    rows = []
    for f in sorted(labels.glob("*.txt"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0)):
        frame = int("".join(c for c in f.stem if c.isdigit()) or 0)
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) >= 6:
                rows.append((frame, float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    out = root / "dets" / f"official_{vid}.txt"
    with out.open("w") as fh:
        for r in sorted(rows):
            fh.write("%d %.6f %.6f %.6f %.6f %.4f\n" % r)
    print("wrote", out, len(rows), "rows")
PY
  cd "$WORK" && tar czf official_dets.tgz dets eval && echo "package: $WORK/official_dets.tgz"
fi

echo "== done. send back $WORK/official_dets.tgz (if produced) and this log."
