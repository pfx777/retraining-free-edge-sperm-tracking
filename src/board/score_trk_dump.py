#!/usr/bin/env python3
"""Score the board's *own tracker output* against ground truth.

Every MOTA this project quoted before 2026-09-23 came from replaying a board
detection dump through the board's tracker class on the host
(`score_det_dump.py`, `score_board_dump_gctx.py`, `board_eval_scales.py`).  That
is a simulation of the board, not the board.  `rknn_headraw_pipeline.py
--dump-dir` writes what the tracker actually produced on the RK3568, one file
per frame:

    <frame>.txt   lines: 0 <cx> <cy> <w> <h> <track_id> <conf>   (normalised)

This script reads that directory and computes MOTA with exactly the protocol
used everywhere else in the project (class-0 ground truth, IoU>=0.5 greedy
matching, ID switches charged to the GT's previously matched track id, gaps up
to `--gap` linearly interpolated).

Note the semantics of `--box-scale` on the board: `rknn_headraw_pipeline.py`
multiplies **only the reported** w/h (matching `run_tracking_deepocsort.py`), so
the tracker matched on unscaled boxes.  The host harnesses instead scale the
detections before the tracker.  `board_trk_replay.py` separates those two.

    python3 score_trk_dump.py --trk-root .tmp/trk_pruned --vids 82,60,54,52
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score_det_dump import interpolate, mota, parse_mot  # noqa: E402


def find_trk_dir(root: Path, v: str) -> Path | None:
    for name in (f"trk_pruned_{v}", f"trk_{v}", f"trk4_{v}", v):
        p = root / name
        if p.is_dir() and any(p.glob("*.txt")):
            return p
    return None


def read_trk(path: Path) -> tuple[dict[int, dict[int, tuple]], int]:
    """-> ({track_id: {frame: (cx, cy, w, h)}}, max_frame)"""
    tracks: dict[int, dict[int, tuple]] = defaultdict(dict)
    max_frame = 0
    for f in path.glob("*.txt"):
        try:
            fr = int(f.stem)
        except ValueError:
            continue
        max_frame = max(max_frame, fr)
        for line in f.read_text().splitlines():
            q = line.split()
            if len(q) < 6:
                continue
            # 0 cx cy w h id conf -- the leading 0 is the class, id is column 5
            cx, cy, w, h = (float(q[1]), float(q[2]), float(q[3]), float(q[4]))
            tid = int(float(q[5]))
            tracks[tid][fr] = (cx, cy, w, h)
    return dict(tracks), max_frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trk-root", required=True,
                    help="directory holding one subdir per session, each with <frame>.txt")
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--gt-dir", default=str(HERE / "datasets" / "visem_tracking" / "mot_gt_c0"))
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=0, help="0 = use each dump's max frame")
    ap.add_argument("--gap", type=int, default=14)
    ap.add_argument("--no-interp", action="store_true",
                    help="score the raw per-frame dump without gap filling")
    args = ap.parse_args()

    root = Path(args.trk_root)
    vids = [v for v in args.vids.split(",") if v]
    print(f"{'vid':>4s} {'frames':>7s} {'tracks':>7s} {'rows':>8s} {'MOTA':>8s} "
          f"{'FP':>7s} {'FN':>7s} {'IDSW':>6s} {'GT':>7s}")
    per_vid = {}
    for v in vids:
        d = find_trk_dir(root, v)
        if d is None:
            print(f"{v:>4s}  no tracker dump under {root}, skipped")
            continue
        tracks, max_frame = read_trk(d)
        if not tracks:
            print(f"{v:>4s}  tracker dump empty, skipped")
            continue
        hi = args.hi if args.hi > 0 else max_frame
        hi = min(hi, max_frame)
        tracks = {t: {f: b for f, b in frs.items() if args.lo <= f <= hi}
                  for t, frs in tracks.items()}
        tracks = {t: b for t, b in tracks.items() if b}
        gt = parse_mot(Path(args.gt_dir) / f"{v}.txt")
        gt = {f: b for f, b in gt.items() if args.lo <= f <= hi}
        if args.no_interp:
            raw: dict[int, list] = defaultdict(list)
            for tid, frs in tracks.items():
                for f, b in frs.items():
                    raw[f].append((tid, b))
            pred = dict(raw)
        else:
            pred = interpolate(tracks, args.gap)
        m = mota(gt, pred)
        per_vid[v] = m["MOTA"]
        n_rows = sum(len(b) for b in tracks.values())
        print(f"{v:>4s} {hi:7d} {len(tracks):7d} {n_rows:8d} {m['MOTA']:8.4f} "
              f"{m['FP']:7d} {m['FN']:7d} {m['IDSW']:6d} {m['GT']:7d}")

    if per_vid:
        vals = np.asarray(list(per_vid.values()))
        print(f"\nboard tracker, four-session mean MOTA = {vals.mean():.4f} "
              f"(min {vals.min():.4f}, max {vals.max():.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
