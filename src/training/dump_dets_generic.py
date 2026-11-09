#!/usr/bin/env python3
"""Produce the res_pareto detection cache with a plain, memory-safe loop.

`res_pareto.infer` uses `model.predict(stream=True, batch=N)`.  For the P2 arm
(`v8n_p2_448`, four detection levels, stride 4 at the finest) that path allocates
a multi-GB tensor and OOMs even at batch=1 -- while a single-image predict uses
40 MB.  This script writes the same cache files (`dets_<vid>_<size>.npy`) with an
explicit chunked loop and `stream=False`, after which `res_pareto.py` finds the
cache and only does the scoring (it skips inference when the file exists).

    python3 dump_dets_generic.py --model M.pt --vids 82,60,54,52 \
        --cache /root/autodl-tmp/resp2b1 --imgsz 448
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# checkpoints of the custom arms (self-calibrating head, cue gain, ...) contain
# modules that only exist after mmsc_modules.register(); without this the
# load fails with "Can't get attribute 'CueGain'".
sys.path.insert(0, "/root/code")
try:
    import mmsc_modules
    mmsc_modules.register()
except Exception:                                          # pragma: no cover
    pass


def session_paths(images_dir: Path, vid: str):
    pairs = []
    for p in images_dir.glob(f"{vid}_frame_*.jpg"):
        pairs.append((int(p.stem.split("_")[-1]), p))
    pairs.sort()
    return [p for _i, p in pairs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="/root/code/datasets/visem_official/val/images")
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    images = Path(args.images)

    for v in [x for x in args.vids.split(",") if x]:
        npy = cache / f"dets_{v}_{args.imgsz}.npy"
        if npy.exists():
            print(f"{v}: cached already ({npy})")
            continue
        paths = session_paths(images, v)
        out = []
        for i in range(0, len(paths), args.batch):
            chunk = [str(p) for p in paths[i:i + args.batch]]
            res = model.predict(source=chunk, imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, max_det=args.max_det, verbose=False,
                                stream=False, batch=args.batch, device=args.device)
            for r in res:
                h, w = r.orig_shape
                b = r.boxes
                if b is None or len(b) == 0:
                    out.append(np.zeros((0, 5), np.float32))
                    continue
                xy = b.xyxy.cpu().numpy()
                c = b.conf.cpu().numpy().astype(np.float32)
                out.append(np.stack([(xy[:, 0] + xy[:, 2]) / 2 / w,
                                     (xy[:, 1] + xy[:, 3]) / 2 / h,
                                     (xy[:, 2] - xy[:, 0]) / w,
                                     (xy[:, 3] - xy[:, 1]) / h, c], 1
                                    ).astype(np.float32))
        np.save(npy, np.array(out, dtype=object), allow_pickle=True)
        print(f"{v}: {len(out)} frames, {sum(len(x) for x in out)} detections -> {npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
