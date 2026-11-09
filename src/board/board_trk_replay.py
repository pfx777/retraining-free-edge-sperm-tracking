#!/usr/bin/env python3
"""Compare three ways of getting "the board's MOTA" for the same session.

  A  host replay, scale applied to the *detections* before the tracker
     (what `score_board_dump_gctx.py` / `board_eval_scales.py` do)      -> the
     number that was being quoted as "board MOTA"
  B  host replay, scale applied to the *reported* boxes only after the tracker
     (what `rknn_headraw_pipeline.py` and `run_tracking_deepocsort.py` actually
     do on the board)
  C  the board's own tracker dump (`score_trk_dump.py`)

A and B differ only in whether the per-session scale enters the association
step.  C should track B if the host replay is a faithful simulation; the gap
between B and C is the part of "board MOTA" that was never board.

All three are scored on the same frame range (default 1..200, the range the
board tracker dump covers) with identical ground truth and MOTA protocol.

    python3 board_trk_replay.py --dets-root .tmp/pruned_4sess \
        --trk-root .tmp/trk_pruned --scales 82:1.02,60:1.15,54:1.48,52:1.16
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adaptive_sched_eval import Replay, load_board_tracker      # noqa: E402
from score_board_dump_gctx import find_dump, read_board          # noqa: E402
from score_det_dump import interpolate, mota, parse_mot           # noqa: E402
from score_trk_dump import find_trk_dir, read_trk                 # noqa: E402

FW, FH = 640.0, 480.0


def scale_tracks(tracks, s):
    return {t: {f: (b[0], b[1], b[2] * s, b[3] * s) for f, b in frs.items()}
            for t, frs in tracks.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets-root", required=True)
    ap.add_argument("--trk-root", required=True)
    ap.add_argument("--scales", required=True, help="e.g. 82:1.02,60:1.15")
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--gt-dir", default=str(HERE / "datasets/visem_tracking/mot_gt_c0"))
    ap.add_argument("--board-script", default=str(HERE / "rknn_headraw_pipeline.py"))
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--frames", type=int, default=200,
                    help="how many frames to score; 0 = every frame in the detection dump")
    ap.add_argument("--stride", type=int, default=1,
                    help="emulate the board running its detector every k-th frame "
                         "(the tracker coasts in between)")
    ap.add_argument("--gap", type=int, default=14)
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--low-match-iou", type=float, default=0.70)
    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--track-high", type=float, default=0.60)
    ap.add_argument("--new-track", type=float, default=0.60)
    ap.add_argument("--track-low", type=float, default=0.05)
    ap.add_argument("--stats", action="store_true",
                    help="print the full B/C confusion counts and whether B == C")
    args = ap.parse_args()

    sc = {}
    for item in args.scales.split(","):
        k, v = item.split(":")
        sc[k.strip()] = float(v)

    Tracker = load_board_tracker(Path(args.board_script))
    vids = [v for v in args.vids.split(",") if v]
    rows = {}
    print(f"{'vid':>4s} {'frames':>7s} {'scale':>6s} {'A inp-scale':>11s} "
          f"{'B out-scale':>11s} {'C board trk':>11s} {'B-C':>8s}")
    for v in vids:
        trk_dir = find_trk_dir(Path(args.trk_root), v)
        board_tracks, max_frame = read_trk(trk_dir) if trk_dir else ({}, 0)
        p = find_dump(Path(args.dets_root), v)
        pf, _r = read_board(p, FW, FH)
        full = max(pf) if pf else 0
        if args.frames > 0:
            hi = min(args.frames, full)
        else:
            hi = full
        gt = parse_mot(Path(args.gt_dir) / f"{v}.txt")
        gt = {f: b for f, b in gt.items() if args.lo <= f <= hi}

        k = max(1, args.stride)
        dec = (lambda st, fr: True) if k == 1 else (lambda st, fr, k=k: (fr - 1) % k == 0)

        # A: scale the detections, then track
        rep = Replay(Tracker, args, pf, dec, FW, FH, emit_age=0)
        tr_a, _a, _b = rep.run(args.lo, hi, v, sc[v])
        m_a = mota(gt, interpolate(tr_a, args.gap))["MOTA"]

        # B: track on unscaled detections, then scale the reported boxes
        rep = Replay(Tracker, args, pf, dec, FW, FH, emit_age=0)
        tr_b, _a, _b = rep.run(args.lo, hi, v, 1.0)
        m_b = mota(gt, interpolate(scale_tracks(tr_b, sc[v]), args.gap))["MOTA"]

        # C: the board's own tracker dump
        bt = {t: {f: b for f, b in frs.items() if args.lo <= f <= hi}
              for t, frs in board_tracks.items()}
        bt = {t: b for t, b in bt.items() if b}
        covered = max_frame >= hi and max_frame > 0
        m_c_stats = mota(gt, interpolate(bt, args.gap)) if covered else None
        m_c = m_c_stats["MOTA"] if m_c_stats else float("nan")
        m_b_stats = mota(gt, interpolate(scale_tracks(tr_b, sc[v]), args.gap))

        rows[v] = (m_a, m_b, m_c)
        c_txt = f"{m_c:11.4f}" if covered else f"{'n/a':>11s}"
        d_txt = f"{m_b - m_c:+8.4f}" if covered else f"{'(no board dump)':>14s}"
        print(f"{v:>4s} {hi:7d} {sc[v]:6.2f} {m_a:11.4f} {m_b:11.4f} {c_txt} {d_txt}")
        if args.stats:
            for tag, st in (("B", m_b_stats), ("C", m_c_stats)):
                if st is None:
                    continue
                print(f"      {tag}: TP {st['TP']} FP {st['FP']} FN {st['FN']} "
                      f"IDSW {st['IDSW']}")
            if m_c_stats is not None:
                same = all(m_b_stats[k] == m_c_stats[k] for k in ("TP", "FP", "FN", "IDSW"))
                print(f"      B and C confusion counts identical: {same}")

    if rows:
        arr = np.asarray([rows[v] for v in rows])
        c_mean = (f"{arr[:, 2].mean():.4f}" if np.isfinite(arr[:, 2]).all()
                  else "n/a (board dump covers only part of the range)")
        print(f"\nfour-session mean:  A {arr[:, 0].mean():.4f}   "
              f"B {arr[:, 1].mean():.4f}   C {c_mean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
