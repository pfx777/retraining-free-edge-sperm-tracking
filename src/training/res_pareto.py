#!/usr/bin/env python3
"""Accuracy side of the resolution Pareto, for the *delivered* stock v8n.

Deployment rate has two knobs: how often the detector runs (k, already
characterised on the board: 13 -> 53 FPS with a MOTA cost that the stride-aware
tracker thresholds now largely absorb) and **how big an input it runs on**.  This
script produces the accuracy axis of the second knob, using the exact weights
that were deployed (`runs/.../stock_yolov8n_448/weights/best.pt`) evaluated at
several input sizes -- no retraining, so the comparison isolates resolution.

Protocol, per (imgsz, session):
  * one inference pass per frame at conf 0.02 / IoU 0.5 (the deployment floor);
  * the per-session box scale is swept over 0.60..1.60 step 0.05 and the best is
    reported (the same grid the project's pvscale sweeps use);
  * MOTA is reported twice: `full` (oracle scale, whole session) and `calib`
    (the paper protocol -- scale picked on the first 5% of even frames, scored on
    the remaining frames).

    python3 res_pareto.py --model /root/autodl-tmp/.../best.pt \
        --sizes 320,384,448,512 --out runs/visem_tracking/res_pareto.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fasttrack import TrackerArrays                      # noqa: E402
from score_det_dump import interpolate, mota, parse_mot  # noqa: E402


def session_paths(images_dir: Path, vid: str):
    """val images of one session, in the MOT frame order (frame_N -> N+1)."""
    pairs = []
    for p in images_dir.glob(f"{vid}_frame_*.jpg"):
        pairs.append((int(p.stem.split("_")[-1]), p))
    pairs.sort()
    return [p for _i, p in pairs]


def infer(model, paths, imgsz, conf, iou, max_det, batch):
    out = []
    t0 = time.time()
    for r in model.predict(source=[str(p) for p in paths], imgsz=imgsz, conf=conf,
                           iou=iou, max_det=max_det, verbose=False, stream=True,
                           batch=batch):
        h, w = r.orig_shape
        b = r.boxes
        if b is None or len(b) == 0:
            out.append(np.zeros((0, 5), np.float32))
            continue
        xy = b.xyxy.cpu().numpy()
        c = b.conf.cpu().numpy().astype(np.float32)
        row = np.stack([(xy[:, 0] + xy[:, 2]) / 2 / w, (xy[:, 1] + xy[:, 3]) / 2 / h,
                        (xy[:, 2] - xy[:, 0]) / w, (xy[:, 3] - xy[:, 1]) / h, c], 1)
        out.append(row.astype(np.float32))
    print(f"    infer imgsz {imgsz}: {len(out)} frames in {time.time() - t0:.1f}s")
    return out


def replay(dets, scale, args, fw, fh):
    tr = TrackerArrays(args.track_high, args.new_track, args.track_low,
                       args.match_iou, args.low_match_iou, args.max_age)
    tracks = defaultdict(dict)
    empty4 = np.zeros((0, 4), np.float32)
    empty1 = np.zeros((0,), np.float32)
    for i, d in enumerate(dets, start=1):
        if d.shape[0]:
            xywh = np.empty((d.shape[0], 4), np.float32)
            xywh[:, 0] = d[:, 0] * fw
            xywh[:, 1] = d[:, 1] * fh
            xywh[:, 2] = d[:, 2] * fw * scale
            xywh[:, 3] = d[:, 3] * fh * scale
            tr.update(xywh, np.ascontiguousarray(d[:, 4]))
        else:
            tr.update(empty4, empty1)
        for k in range(len(tr.xywh)):
            if tr.age[k] != 0:
                continue
            cx, cy, w, h = tr.xywh[k]
            tracks[int(tr.ids[k])][i] = (cx / fw, cy / fh, w / fw, h / fh)
    return tracks


def subset(tracks, frames):
    keep = set(frames)
    return {tid: {f: b for f, b in v.items() if f in keep} for tid, v in tracks.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="/root/code/datasets/visem_official/val/images")
    ap.add_argument("--gt-dir", default="/root/code/datasets/visem_tracking/mot_gt_c0")
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--sizes", default="320,384,448,512")
    ap.add_argument("--scales", default="0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,"
                                        "1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.35,"
                                        "1.40,1.45,1.50,1.55,1.60")
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fw", type=float, default=640.0)
    ap.add_argument("--fh", type=float, default=480.0)
    ap.add_argument("--gap", type=int, default=14)
    ap.add_argument("--track-high", type=float, default=0.60)
    ap.add_argument("--new-track", type=float, default=0.60)
    ap.add_argument("--track-low", type=float, default=0.05)
    ap.add_argument("--match-iou", type=float, default=0.20)
    ap.add_argument("--low-match-iou", type=float, default=0.70)
    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--cache", default="/root/autodl-tmp/res_dets",
                    help="where to keep the raw detections so the sweep can be re-run")
    ap.add_argument("--dump-only", action="store_true",
                    help="only run the detector and cache the detections (no scoring; "
                         "used to prepare sessions whose MOT ground truth is not built yet)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fixed-scales", default=None,
                    help="e.g. 82:0.90,60:1.40,54:1.50,52:0.95 -- skip the sweep and use "
                         "the deployed per-session scale, so imgsz is the only variable")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    vids = [v for v in args.vids.split(",") if v]
    scales = [float(x) for x in args.scales.split(",")]
    images = Path(args.images)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    rows = []
    fixed = {}
    if args.fixed_scales:
        for item in args.fixed_scales.split(","):
            k_, v_ = item.split(":")
            fixed[k_.strip()] = float(v_)
    for size in [int(x) for x in args.sizes.split(",")]:
        for v in vids:
            npy = cache / f"dets_{v}_{size}.npy"
            if npy.exists():
                dets = list(np.load(npy, allow_pickle=True))
                print(f"    cached {npy.name}: {len(dets)} frames")
            else:
                dets = infer(model, session_paths(images, v), size, args.conf, args.iou,
                             args.max_det, args.batch)
                np.save(npy, np.array(dets, dtype=object), allow_pickle=True)
            if args.dump_only:
                continue
            gt = parse_mot(Path(args.gt_dir) / f"{v}.txt")
            n_frames = len(dets)
            gt = {f: b for f, b in gt.items() if 1 <= f <= n_frames}
            calib = [f for f in range(1, max(2, int(0.05 * n_frames)) + 1) if f % 2 == 0]
            held = [f for f in range(1, n_frames + 1) if f not in set(calib)]

            # The reported number must not pick its scale on the frames it is
            # scored on, or it is an oracle upper bound rather than a deployable
            # protocol.  So: choose the scale on the calibration frames (first 5%
            # of even frames), report the held-out frames at that scale.
            gt_c = {f: b for f, b in gt.items() if f in set(calib)}
            gt_h = {f: b for f, b in gt.items() if f in set(held)}
            if v in fixed:
                tr = replay(dets, fixed[v], args, args.fw, args.fh)
                m_full = mota(gt, interpolate(tr, args.gap))["MOTA"]
                m_held = mota(gt_h, interpolate(subset(tr, held), args.gap))["MOTA"]
                rows.append(dict(imgsz=size, vid=v, scale=fixed[v], mota_calib=m_held,
                                 mota_oracle=m_full, oracle_scale=fixed[v], mota_full=m_full))
                print(f"  imgsz {size:4d} vid {v}: fixed scale {fixed[v]:.2f}  "
                      f"MOTA_full {m_full:.4f}  held-out {m_held:.4f}")
                continue
            best_cal, best_cal_scale = None, None
            best_oracle, best_oracle_scale, oracle_full = None, None, None
            for sc in scales:
                tr = replay(dets, sc, args, args.fw, args.fh)
                m_cal = mota(gt_c, interpolate(subset(tr, calib), args.gap))["MOTA"]
                if best_cal is None or m_cal > best_cal:
                    best_cal, best_cal_scale = m_cal, sc
                m_full = mota(gt, interpolate(tr, args.gap))["MOTA"]
                if best_oracle is None or m_full > best_oracle:
                    best_oracle, best_oracle_scale = m_full, sc
            tr = replay(dets, best_cal_scale, args, args.fw, args.fh)
            held_mota = mota(gt_h, interpolate(subset(tr, held), args.gap))["MOTA"]
            rows.append(dict(imgsz=size, vid=v, scale=best_cal_scale,
                             mota_calib=held_mota, mota_oracle=best_oracle,
                             oracle_scale=best_oracle_scale, mota_full=held_mota))
            print(f"  imgsz {size:4d} vid {v}: 5%-->{best_cal_scale:.2f} MOTA {held_mota:.4f} "
                  f"| oracle {best_oracle_scale:.2f} {best_oracle:.4f}")

    print()
    print(f"{'size':>6s} {'5%protocol':>10s} {'oracle':>8s}   per-session (5% protocol)")
    for size in sorted({r["imgsz"] for r in rows}):
        rs = [r for r in rows if r["imgsz"] == size]
        print(f"{size:6d} {np.mean([r['mota_calib'] for r in rs]):10.4f} "
              f"{np.mean([r['mota_oracle'] for r in rs]):8.4f}   "
              + " ".join(f"{r['mota_calib']:8.4f}" for r in rs))

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["imgsz", "vid", "scale", "mota_calib",
                                               "mota_oracle", "oracle_scale", "mota_full"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
