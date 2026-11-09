#!/bin/bash
# E26 fourth seed (2024) -- same reason as E26s123: pin the mean of the headline
# module's main-table gain (seed42 +0.1056, seed7 +0.0557 so far).
set -u
cd /root/code
export PATH=/root/miniconda3/bin:$PATH

while [ ! -f logs/E26s123_done.txt ]; do sleep 120; done
echo "[E26s2024] queue clear, starting $(date)"

bash /root/code/run_arch_arm2.sh v8n_gctx448_s2024 /root/code/stock_yolov8n_gctx.yaml E26s2024 2024
echo "[E26s2024] chain done $(date)"
