#!/usr/bin/env python3
"""Is it the board's tracker or the board's INT8 detector that degrades late in a
session?

On video 82 the host loses 0.022 MOTA from frames 1-300 to 301-1500 while the
board loses 0.110, and the host's precision improves over that stretch (0.917 ->
0.929) while the board's falls (0.881 -> 0.834).  Frame statistics rule out a
harder second half (brightness 90.15 vs 90.00, Laplacian 3.76 vs 3.75, GT/frame
28.3 vs 31.3).

So the same cached detections are replayed through BOTH trackers here -- the board
one, sliced out of rknn_headraw_pipeline.py, and the host one -- and scored
segment by segment.  If the board tracker also collapses in the later segment then
the tracker is at fault; if it tracks the host, the fault is in the INT8
detections and only there.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/root/code")
sys.path.insert(0, str(ROOT))


def load_board_tracker(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def iou_xywh"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("def main()"))
    ns: dict = {"np": np}
    exec(compile("".join(lines[start:end]), str(path), "exec"), ns)
    return ns["Tracker"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def scale_boxes(xyxy: np.ndarray, s: float) -> np.ndarray:
    if xyxy.size == 0 or s == 1.0:
        return xyxy
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    w = (xyxy[:, 2] - xyxy[:, 0]) * s
    h = (xyxy[:, 3] - xyxy[:, 1]) * s
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def greedy_nms(boxes, scores, thr):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=bool)
    keep = np.zeros(len(boxes), dtype=bool)
    kept: list[int] = []
    for i in np.argsort(-scores):
        if kept and iou_matrix(boxes[i][None, :], boxes[kept])[0].max() >= thr:
            continue
        keep[i] = True
        kept.append(i)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vid", default="82")
    ap.add_argument("--scale", type=float, default=0.80)
    ap.add_argument("--board-script", default="/root/code/rknn_headraw_pipeline.py")
    ap.add_argument("--conf-thr", type=float, default=0.02)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--segments", default="1:300,301:1500")
    args = ap.parse_args()

    import mmsc_modules  # noqa: F401
    import run_tracking_deepocsort as rtd  # noqa: E402
    import run_tracking_interp_sweep as interp  # noqa: E402

    BoardTracker = load_board_tracker(Path(args.board_script))

    z = np.load(ROOT / "logs" / f"funnel_{args.vid}.npz", allow_pickle=True)
    shapes, boxes, confs = z["shapes"], z["boxes"], z["confs"]
    gt_all = rtd.parse_mot(str(rtd.MOT_GT / f"{args.vid}.txt"))

    # stage-B detections, identical for both trackers
    per_frame = {}
    for fi, ((ih, iw), db, dc) in enumerate(zip(shapes, boxes, confs), start=1):
        db = np.asarray(db, dtype=np.float32)
        dc = np.asarray(dc, dtype=np.float32)
        m = dc >= args.conf_thr
        db, dc = db[m], dc[m]
        if len(db):
            keep = greedy_nms(db, dc, args.nms_iou)
            db, dc = db[keep], dc[keep]
        per_frame[fi] = (scale_boxes(db, args.scale), dc, ih, iw)

    out = {}
    for name in ("board", "host"):
        if name == "board":
            tracker = BoardTracker(0.60, 0.60, 0.05, 0.30, 0.70, 60)
        else:
            rtd.TRACK_HIGH, rtd.NEW_TRACK, rtd.TRACK_LOW = 0.60, 0.60, 0.05
            rtd.LOW_MATCH_IOU, rtd.MATCH_IOU = 0.70, 0.30
            rtd.MIN_HITS, rtd.MAX_AGE, rtd.LAMBDA_APP = 1, 60, 0.0
            tracker = rtd.Tracker()
        tracks = defaultdict(dict)
        for fi, (db, dc, ih, iw) in per_frame.items():
            dets = [{"xywh": np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                                       b[2] - b[0], b[3] - b[1]], np.float32),
                     "conf": float(c), "feat": None} for b, c in zip(db, dc)]
            if name == "board":
                tracker.update(dets)
                rows = [(tracker.xywh[k], int(tracker.ids[k]))
                        for k in range(len(tracker.xywh)) if tracker.age[k] == 0]
            else:
                rows = [(t.xywh, int(t.id)) for t in tracker.update(dets)]
            for box, tid in rows:
                tracks[tid][fi] = (box[0] / iw, box[1] / ih, box[2] / iw, box[3] / ih)
        frames = interp.interpolate_tracks(tracks, 14)
        pred = defaultdict(list)
        for fr, items in frames.items():
            for tid, box in items:
                pred[fr].append({"id": tid, "bbox": list(box)})
        out[name] = pred

    print(f"video {args.vid}: SAME cached detections through both trackers")
    print(f"  {'segment':<14}{'board tracker':>16}{'host tracker':>15}{'delta':>9}")
    for spec in args.segments.split(","):
        lo, hi = (int(x) for x in spec.split(":"))
        gt = {f: b for f, b in gt_all.items() if lo <= f <= hi}
        mb = rtd.mot_metrics(gt, {f: v for f, v in out["board"].items() if lo <= f <= hi})["MOTA"]
        mh = rtd.mot_metrics(gt, {f: v for f, v in out["host"].items() if lo <= f <= hi})["MOTA"]
        print(f"  {f'{lo}-{hi}':<14}{mb:>16.4f}{mh:>15.4f}{mb - mh:>+9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
