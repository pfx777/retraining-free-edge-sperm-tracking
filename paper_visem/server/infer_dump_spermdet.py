#!/usr/bin/env python3
"""Same dump format as infer_dump.py, but loading the model from the SpermDet fork.

SpermDet has its own `models` package (custom SAF / LCEB / SDF blocks), so the
checkpoint must be unpickled with that fork on sys.path.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path("/root/autodl-tmp/spermdet_job/SpermDet")
sys.path.insert(0, str(REPO))

from models.experimental import attempt_load          # noqa: E402
from utils.augmentations import letterbox             # noqa: E402
from utils.general import non_max_suppression  # noqa: E402

try:                       # YOLOv5 v7 renamed scale_coords -> scale_boxes
    from utils.general import scale_coords  # type: ignore  # noqa: E402
except ImportError:        # pragma: no cover
    from utils.general import scale_boxes as scale_coords  # type: ignore  # noqa: E402


def frame_index(stem: str) -> int:
    m = re.search(r"frame_(\d+)", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    # fuse=True hits a NameError (DWConv not imported) inside this fork's
    # models/yolo.py:143, so fuse is disabled; inference is unaffected.
    try:
        model = attempt_load(args.weights, device=device, fuse=False)
    except TypeError:
        model = attempt_load(args.weights, map_location=device, fuse=False)
    model.eval()
    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    print(f"model {Path(args.weights).name} on {device}, stride {stride}, names {getattr(model, 'names', None)}", flush=True)

    src = Path(args.src)
    images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    if args.limit:
        images = images[: args.limit]
    print(f"{len(images)} images from {src}", flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, tuple] = {}
    t0 = time.time()
    n_det = 0
    for i, path in enumerate(images, 1):
        vid = path.stem.split("_")[0]
        if vid not in writers:
            writers[vid] = (open(out_dir / f"official_{vid}.txt", "w"),
                            open(out_dir / f"official_all_{vid}.txt", "w"))
        fh, fh_all = writers[vid]
        img0 = cv2.imread(str(path))
        if img0 is None:
            continue
        img, _r, _p = letterbox(img0, args.imgsz, stride=stride, auto=True)
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(img).to(device).float()[None] / 255.0
        with torch.no_grad():
            pred = model(tensor)[0]
            pred = non_max_suppression(pred, args.conf, args.iou, max_det=300)[0]
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(tensor.shape[2:], pred[:, :4], img0.shape).round()
            h, w = img0.shape[:2]
            for x1, y1, x2, y2, conf, cls in pred.tolist():
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                row = f"{frame_index(path.stem)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}"
                fh_all.write(row + f" {int(cls)}\n")
                fh.write(row + "\n")
                n_det += 1
        if i % 500 == 0:
            print(f"  {i}/{len(images)} images, {n_det} detections, {i/(time.time()-t0):.1f} img/s", flush=True)
    for pair in writers.values():
        for fh in pair:
            fh.close()
    print(f"done: {len(images)} images, {n_det} detections, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
