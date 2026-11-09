#!/bin/bash
# Runner v2.  Two lessons are baked in:
#
# 1. `env VAR=... $EXTRA_ENV cmd` is fragile -- when EXTRA_ENV expands, bash has
#    already decided which words were assignments, and with an empty value the
#    line can be re-parsed in surprising ways.  Plain `export` then the command.
# 2. **Never scp over a script while it is running.**  bash reads scripts
#    incrementally by byte offset; overwriting the file mid-run made one chain
#    execute garbage (`e: command not found`) and *truncated and restarted* a
#    training log (E22s7 lost ~2 h of progress).  This is a new file for exactly
#    that reason -- v1 stays untouched.
set -u
cd /root/code
export PATH=/root/miniconda3/bin:$PATH
PY=/root/miniconda3/bin/python3
NAME=$1
CFG=$2
TAG=${3:-$NAME}
SEED=${4:-42}
EXTRA_ENV=${5:-}
BASE_W=/root/autodl-tmp/runs/detect/runs/visem_official
TRAINVIDS=11,12,13,14,15,19,21,22,23,24,29,30,35,36,38,47

fail() { echo "[$TAG] FAIL: $* $(date)"; exit 1; }

# MANDATORY deployment pre-check (learned with EMA): the graph, not the module
# description, decides whether something can ship.  Abort before training if the
# head-raw ONNX contains the CPU-fallback op family.
$PY export_onnx_deploy.py --model "$CFG" --out "/tmp/audit_$TAG.onnx" \
  --imgsz 448 --head-raw --opset 19 --fuse > "logs/${TAG}_audit_export.log" 2>&1
$PY audit_onnx_ops.py --onnx "/tmp/audit_$TAG.onnx" > "logs/${TAG}_onnx_audit.log" 2>&1
# NOTE the pattern: the CLEAN verdict reads "no high-risk op type present", so a
# naive grep for "high-risk op type" matches the passing case too and aborted
# every arm (E25/E26/E27 all "failed" their audit at 15:15).  The failure line is
# distinguishable by the literal "(s)" that the audit prints for plural counts.
if grep -q "high-risk op type(s)" "logs/${TAG}_onnx_audit.log"; then
  echo "[$TAG] ONNX AUDIT FAILED -- not deployable, aborting before training:"
  tail -4 "logs/${TAG}_onnx_audit.log"
  echo DONE > "logs/${TAG}_done.txt"
  exit 0
fi
echo "[$TAG] ONNX audit clean: $(tail -2 logs/${TAG}_onnx_audit.log | head -1)"

export DATA=/root/code/datasets/visem_official/data.yaml
export EPOCHS=30 SAVE_PERIOD=-1 DYHEAD=0 REG=0 BASE=/root/code/yolov8n.pt
export CFG SEED NAME IMGSZ=448 PYTHONUNBUFFERED=1
if [ -n "$EXTRA_ENV" ]; then export $EXTRA_ENV; fi
$PY train_ablation.py > "logs/train_$TAG.log" 2>&1
echo "[$TAG] train rc=$? $(date)"
W=$BASE_W/$NAME/weights/last.pt
[ -f "$W" ] || fail "no last.pt (see logs/train_$TAG.log)"

$PY -u dump_dets_generic.py --model "$W" --vids 82,60,54,52 \
  --cache "/root/autodl-tmp/res_$TAG" --imgsz 448 --batch 8 \
  > "logs/${TAG}_dump_val.log" 2>&1
$PY -u res_pareto.py --model "$W" --sizes 448 --cache "/root/autodl-tmp/res_$TAG" \
  --out "/root/code/runs/visem_tracking/${TAG}_val_scalescan.csv" \
  > "logs/${TAG}_val_scan.log" 2>&1
echo "[$TAG] val scan rc=$? $(date)"

$PY -u dump_dets_generic.py --model "$W" \
  --images /root/code/datasets/visem_official/train/images \
  --vids "$TRAINVIDS" --cache "/root/autodl-tmp/${TAG}_train_dets" --imgsz 448 --batch 8 \
  > "logs/${TAG}_dump_train.log" 2>&1
$PY -u res_pareto.py --model "$W" \
  --images /root/code/datasets/visem_official/train/images \
  --gt-dir /root/code/datasets/visem_tracking/mot_gt_train \
  --vids "$TRAINVIDS" --sizes 448 --scales 1.00 \
  --cache "/root/autodl-tmp/${TAG}_train_dets" \
  --out "/root/code/runs/visem_tracking/${TAG}_train_s1.csv" \
  > "logs/${TAG}_train_s1.log" 2>&1
echo "[$TAG] train s=1 rc=$? $(date)"

$PY -u calibrated_table.py > "logs/${TAG}_main_table.log" 2>&1
echo "[$TAG] main table -> logs/${TAG}_main_table.log"
echo DONE > "logs/${TAG}_done.txt"
echo "[$TAG] all done $(date)"
