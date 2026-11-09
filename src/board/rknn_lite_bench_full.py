#!/usr/bin/env python3
"""
rknn_lite_bench_full.py — RK3568 板端延迟测试（FP16）。

测量：
  - rknn.inference() 纯推理耗时
  - preprocess + inference + output copy 的完整流水线耗时

Usage:
  python3 rknn_lite_bench_full.py \
      --rknn /path/model.rknn \
      --images /path/to/images.txt \
      --runs 100 --warmup 20
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


parser = argparse.ArgumentParser()
parser.add_argument("--rknn", required=True)
parser.add_argument("--images", default="/root/code/runs/deploy/dataset.txt")
parser.add_argument("--size", type=int, default=640)
parser.add_argument("--runs", type=int, default=100)
parser.add_argument("--warmup", type=int, default=20)
args = parser.parse_args()


def letterbox(path, size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dy, dx = (size - nh) // 2, (size - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img
    return canvas


def main():
    paths = [line.strip() for line in open(args.images, encoding="utf-8") if line.strip()]
    if not paths:
        raise SystemExit(f"No images in {args.images}")

    rknn = RKNNLite()
    ret = rknn.load_rknn(args.rknn)
    if ret != 0:
        raise SystemExit(f"load_rknn failed: {ret}")
    attempts = ({"core_mask": 0}, {"target": "rk3568"}, {})
    for kwargs in attempts:
        try:
            ret = rknn.init_runtime(**kwargs)
        except Exception as e:
            print(f"init_runtime {kwargs or 'default'} failed: {e}")
            ret = -1
        if ret == 0:
            print(f"init_runtime OK: {kwargs or 'default'}")
            break
    if ret != 0:
        raise SystemExit(f"init_runtime failed: {ret}")

    # warmup
    for p in paths[: min(args.warmup, len(paths))]:
        x = letterbox(p, args.size)
        x = np.expand_dims(x, 0)
        _ = rknn.inference(inputs=[x])

    npu_times, full_times = [], []
    for p in paths[: min(args.runs, len(paths))]:
        t0 = time.perf_counter()
        x = letterbox(p, args.size)
        x = np.expand_dims(x, 0)
        t1 = time.perf_counter()
        outs = rknn.inference(inputs=[x])
        t2 = time.perf_counter()
        _ = [np.ascontiguousarray(o) for o in outs]
        t3 = time.perf_counter()
        npu_times.append((t2 - t1) * 1000)
        full_times.append((t3 - t0) * 1000)

    npu = np.asarray(npu_times)
    full = np.asarray(full_times)
    print(f"images: {len(npu_times)}  runs: {args.runs}")
    print(f"npU_inference_ms: mean={npu.mean():.3f}  std={npu.std():.3f}")
    print(f"full_pipeline_ms: mean={full.mean():.3f}  std={full.std():.3f}")
    print(f"output_shapes: {[o.shape for o in outs]}")
    rknn.release()


if __name__ == "__main__":
    main()
