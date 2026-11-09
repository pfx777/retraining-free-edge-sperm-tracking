#!/usr/bin/env python3
"""Verify that RKNNLite inference is executing on the RK3568 NPU."""
import argparse
import os
import threading
import time

import numpy as np
from rknnlite.api import RKNNLite


def find_npu_load_path():
    candidates = [
        "/sys/kernel/debug/rknpu/load",
        "/sys/class/devfreq/fde40000.npu/load",
        "/sys/class/devfreq/fde60000.npu/load",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    matches = []
    for root in ["/sys/kernel/debug", "/sys/class/devfreq"]:
        for dirpath, dirnames, filenames in os.walk(root):
            if "npu" in dirpath.lower() or "npu" in ",".join(filenames).lower():
                matches.append(dirpath)
            if len(matches) >= 10:
                break
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    rknn = RKNNLite(verbose=True)
    print("load_rknn:", rknn.load_rknn(args.model))

    # RK3568 has a single NPU core. core_mask=0 selects it explicitly.
    ret = rknn.init_runtime(core_mask=0)
    print("init_runtime ret:", ret)

    load_path = find_npu_load_path()
    print("npu_load_path:", load_path)

    dummy = np.random.randint(0, 255, size=(1, args.imgsz, args.imgsz, 3), dtype=np.uint8)

    stop = False
    loads = []

    def sampler():
        if isinstance(load_path, str):
            while not stop:
                try:
                    with open(load_path, "r") as f:
                        loads.append(f.read().strip())
                except Exception as e:
                    loads.append(f"ERR:{e}")
                    break
                time.sleep(0.05)

    if isinstance(load_path, str):
        t = threading.Thread(target=sampler)
        t.start()

    for _ in range(10):
        _ = rknn.inference(inputs=[dummy])

    t0 = time.perf_counter()
    for _ in range(args.runs):
        _ = rknn.inference(inputs=[dummy])
    dt_ms = (time.perf_counter() - t0) / args.runs * 1000

    stop = True
    if isinstance(load_path, str):
        t.join()

    print(f"latency_ms: {dt_ms:.3f}  runs={args.runs}")
    print("sampled_npu_loads:", loads[:40])
    print("sampled_count:", len(loads))

    rknn.release()


if __name__ == "__main__":
    main()
