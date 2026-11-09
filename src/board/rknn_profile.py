#!/usr/bin/env python3
"""
rknn_profile.py — 在转换端做逐层耗时分析，定位 CPU fallback 算子。

Usage:
  python rknn_profile.py \
      --onnx /root/code/runs/deploy/legacy_head_v1.onnx \
      --target rk3568
"""

import argparse

import numpy as np
from rknn.api import RKNN


parser = argparse.ArgumentParser()
parser.add_argument("--onnx", required=True)
parser.add_argument("--target", default="rk3568")
parser.add_argument("--input-size", type=int, default=640)
parser.add_argument("--sim", action="store_true", help="在 PC 模拟器上 profile，不连接实体板卡")
args = parser.parse_args()

rknn = RKNN(verbose=True)
ret = rknn.config(
    target_platform=args.target,
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    optimization_level=3,
)
assert ret == 0, f"config failed: {ret}"

ret = rknn.load_onnx(model=args.onnx, inputs=["images"], input_size_list=[[1, 3, args.input_size, args.input_size]])
assert ret == 0, f"load_onnx failed: {ret}"

ret = rknn.build(do_quantization=False)
assert ret == 0, f"build failed: {ret}"

if args.sim:
    ret = rknn.init_runtime(perf_debug=True)
else:
    ret = rknn.init_runtime(target=args.target, perf_debug=True)
assert ret == 0, f"init_runtime failed: {ret}"

x = np.full((1, args.input_size, args.input_size, 3), 128, dtype=np.uint8)
outs = rknn.inference(inputs=[x], data_format="nhwc")
print(f"output shapes: {[o.shape for o in outs]}")

try:
    perf = rknn.eval_perf(is_print=True)
    print("Per-layer profile:", perf)
except Exception as e:
    print("eval_perf not available:", e)

rknn.release()
