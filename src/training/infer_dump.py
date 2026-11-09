#!/usr/bin/env python3
"""Run an official VISEM-Tracking YOLOv5 checkpoint and dump MOT-format detections.

Uses only the fork's model loader, letterbox and NMS (the paths that are numpy-2
safe); no training/plotting code is touched.  Output rows are

    frame cx cy w h score          (coordinates normalised to the original frame)

so the host-side scorer and the calibration fitter can consume them directly.

    python infer_dump.py --weights ckpt/yolov5x_best.pt --src <images dir> \
        --out dets/yolov5x --conf 0.02 --iou 0.5 --imgsz 640
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

REPO = Path("/root/autodl-tmp/visem_official_job/visem-tracking/sample_YOLO_models")
sys.path.insert(0, str(REPO))

from models.experimental import attempt_load          # noqa: E402
from utils.augmentations import letterbox             # noqa: E402
from utils.general import non_max_suppression, scale_coords  # noqa: E402


def load_model(weights: Path, device: str):
    try:
        model = attempt_load(str(weights), device=device, fuse=True)
    except TypeError:
        model = attempt_load(str(weights), map_location=device, fuse=True)
    model.eval()
    return model


def frame_index(stem: str) -> int:
    m = re.search(r"frame_(\d+)", stem)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", required=True, help="directory of images (one video or many)")
    ap.add_argument("--out", required=True, help="output directory; one .txt per video")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--classes", default="0",
                    help="comma-separated class ids to keep in the class-0 file; "
                         "'all' writes every class into the *_all file")
    args = ap.parse_args()
    keep = None if args.classes == "all" else {int(c) for c in args.classes.split(",") if c != ""}

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.weights), device)
    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    print(f"model {Path(args.weights).name} on {device}, stride {stride}, "
          f"names {getattr(model, 'names', None)}", flush=True)

    src = Path(args.src)
    images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    if args.limit:
        images = images[: args.limit]
    print(f"{len(images)} images from {src}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, "open"] = {}
    writer_for: dict[str, object] = {}
    t0 = time.time()
    n_det = 0
    for i, path in enumerate(images, 1):
        vid = path.stem.split("_")[0]
        if vid not in writer_for:
            writer_for[vid] = (
                open(out_dir / f"official_{vid}.txt", "w"),
                open(out_dir / f"official_all_{vid}.txt", "w"),
            )
        fh, fh_all = writer_for[vid]

        img0 = cv2.imread(str(path))
        if img0 is None:
            continue
        img, ratio, pad = letterbox(img0, args.imgsz, stride=stride, auto=True)
        img = img[:, :, ::-1].transpose(2, 0, 1)          # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(device).float() / 255.0
        tensor = tensor[None]

        with torch.no_grad():
            pred = model(tensor)[0]
            pred = non_max_suppression(pred, args.conf, args.iou, max_det=300)[0]
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(tensor.shape[2:], pred[:, :4], img0.shape).round()
            h, w = img0.shape[:2]
            for x1, y1, x2, y2, conf, cls in pred.tolist():
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                row = f"{frame_index(path.stem)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}"
                fh_all.write(row + f" {int(cls)}\n")
                if keep is None or int(cls) in keep:
                    fh.write(row + "\n")
                    n_det += 1
        if i % 500 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(images)} images, {n_det} detections, {i/el:.1f} img/s", flush=True)
    for pair in writer_for.values():
        for fh in pair:
            fh.close()
    print(f"done: {len(images)} images, {n_det} detections, {time.time()-t0:.0f}s", flush=True)
    for vid in sorted(writer_for):
        p = out_dir / f"official_{vid}.txt"
        print("  ", p.name, sum(1 for _ in open(p)), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
