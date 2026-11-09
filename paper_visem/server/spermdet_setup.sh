#!/usr/bin/env bash
# Make the SpermDet fork runnable here and prepare our official split, then smoke test.
set -uo pipefail
R=/root/autodl-tmp/spermdet_job/SpermDet
PY=/root/miniconda3/bin/python
cd "$R" || exit 1

echo "=== 1. shim the one legacy import (ultralytics.yolo.utils.checks)"
mkdir -p ultralytics/yolo/utils
: > ultralytics/__init__.py
: > ultralytics/yolo/__init__.py
: > ultralytics/yolo/utils/__init__.py
cat > ultralytics/yolo/utils/checks.py <<'PY'
"""Minimal stand-in for the pre-2023 ultralytics helper used by this fork.

The fork was written against `ultralytics` <8.1 and only calls
`check_requirements(...)` to install missing pip deps.  Dependencies here are
already satisfied, so this returns True without touching the environment.
"""
from __future__ import annotations


def check_requirements(requirements=(), exclude=(), install=False):
    return True


def check_version(current="0.0.0", minimum="0.0.0", **kwargs):
    return True
PY

echo "=== 2. numpy2 compatibility (np.int -> int) in the fork's utils/models"
sed -i 's/\.astype(np\.int)/.astype(int)/g; s/np\.int\b/int/g; s/np\.float\b/float/g' utils/*.py models/*.py 2>/dev/null
echo "patched"

echo "=== 3. our official split, single class (matches this paper's class-0 protocol)"
cat > data/VISEM_official_c0.yaml <<'YML'
train: /root/code/datasets/visem_official/train/images
val: /root/code/datasets/visem_official/val/images
nc: 1
names: ['sperm']
YML
cat data/VISEM_official_c0.yaml

echo "=== 4. model instantiation + parameter count"
$PY - <<'PY'
import sys
sys.path.insert(0, ".")
from models.yolo import Model
m = Model("models/SpermDet.yaml", ch=3, nc=1)
n = sum(p.numel() for p in m.parameters())
print("SpermDet params %.3f M" % (n / 1e6))
PY

echo "=== 5. 1-epoch smoke test (measures s/epoch; writes to runs_smoke)"
$PY train.py --img 640 --batch 16 --epochs 1 --data data/VISEM_official_c0.yaml \
  --cfg models/SpermDet.yaml --weights '' --workers 8 --device 0 --nosave \
  --project /root/autodl-tmp/spermdet_job/runs_smoke --name smoke --exist-ok 2>&1 | tail -25
echo "=== disk"
df -h /root/autodl-tmp | tail -1
