#!/usr/bin/env python3
"""Dump MOT-format detections for a stock-family checkpoint.

The host-side scorer used for the cross-family table (`official_compare.py`)
reads rows of

    frame cx cy w h score        (coordinates normalised to the original frame)

so this writes exactly that, with the same inference settings as the rest of
the paper (imgsz 448, conf 0.02, iou 0.5, class 0).

    python dump_family_mot.py --weights <run>/weights/last.pt \
        --src /root/code/datasets/visem_official/val/images \
        --out /root/autodl-tmp/fam_dets/val/stock_yolo11n_448
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/code")
try:                                     # custom arms register extra modules
    import mmsc_modules                    # noqa: E402
    mmsc_modules.register()
except Exception:                          # pragma: no cover
    pass

from ultralytics import YOLO               # noqa: E402


def frame_index(stem: str) -> int:
    tail = stem.split("_")[-1]
    return int(tail) if tail.isdigit() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512,
                    help="predict the path list in chunks; a single long list "
                         "is treated as one image sequence and blows up the GPU")
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = YOLO(args.weights)
    images = sorted(Path(args.src).glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    handles: dict[str, object] = {}
    n_det, t0 = 0, time.time()
    # a list source makes ultralytics label results "imageN.jpg", so pair the
    # results back with the input paths instead of reading r.path
    for start in range(0, len(images), args.chunk):
        chunk = images[start:start + args.chunk]
        stream = model.predict(source=[str(p) for p in chunk], imgsz=args.imgsz,
                               conf=args.conf, iou=args.iou, max_det=args.max_det,
                               device=args.device, verbose=False, stream=True,
                               batch=args.batch)
        for r, src_path in zip(stream, chunk):
            vid = src_path.stem.split("_")[0]
            fh = handles.get(vid)
            if fh is None:
                fh = handles[vid] = open(out_dir / f"official_{vid}.txt", "w")
            h, w = r.orig_shape[:2]
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                fh.write(f"{frame_index(src_path.stem)} "
                         f"{(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} "
                         f"{float(b.conf[0]):.4f}\n")
                n_det += 1
        done = min(start + args.chunk, len(images))
        el = time.time() - t0
        print(f"  {done}/{len(images)} images, {n_det} dets, {done / el:.1f} img/s",
              flush=True)
    for fh in handles.values():
        fh.close()
    print(f"done: {len(images)} images, {n_det} detections, "
          f"{time.time() - t0:.0f}s", flush=True)
    for vid in sorted(handles):
        p = out_dir / f"official_{vid}.txt"
        print("  ", p.name, sum(1 for _ in open(p)), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
