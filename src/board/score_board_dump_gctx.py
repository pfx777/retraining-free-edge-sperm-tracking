#!/usr/bin/env python3
"""Score the board's INT8 detection dump against the host FP32 cache.

Inputs:
  --dump-root DIR   dir holding dets_gctx_<vid>/detections.txt (board INT8),
                    lines: <frame> <cx> <cy> <w> <h> <score> <level> normalised
  --host-cache DIR  this arm's host FP32 cache dets_<vid>_448.npy

Reports per session and averaged:
  1. MOTA under the paper protocol (kappa rule lambda=0.5 scales) plus a scale
     sweep, so both the delivered point and the board optimum are visible;
  2. INT8-vs-FP32 detection statistics: score and box-width quantiles;
  3. level-0 (stride-8) box size, the level RKNN is known to distort and the
     one the new gate sits on.

No inference is run: the board dump and the host cache are existing artefacts.

    python3 score_board_dump_gctx.py --dump-root <dir> --host-cache <dir>
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adaptive_sched_eval import Replay, load_board_tracker  # noqa: E402
from score_det_dump import interpolate, mota, parse_mot      # noqa: E402

RULES5 = {"82": 0.95, "60": 1.05, "54": 1.20, "52": 1.05}
SWEEP = [round(0.60 + 0.05 * i, 2) for i in range(21)]


def read_board(path: Path, fw: float, fh: float):
    per_frame = defaultdict(list)
    rows = []
    for line in path.read_text().splitlines():
        q = line.split()
        if len(q) < 6:
            continue
        fr = int(float(q[0]))
        cx, cy, w, h, s = (float(q[1]), float(q[2]), float(q[3]),
                           float(q[4]), float(q[5]))
        lv = int(float(q[6])) if len(q) > 6 else 0
        per_frame[fr].append((lv, s, (cx * fw, cy * fh, w * fw, h * fh)))
        rows.append((cx, cy, w, h, s, lv, fr))
    arr = np.asarray(rows, np.float64) if rows else np.zeros((0, 7))
    return per_frame, arr


def find_dump(dump_root: Path, v: str):
    """Accept either the board naming (dets_gctx_<v>) or the plain one."""
    for name in (f"dets_gctx_{v}", f"dets_{v}", v):
        p = dump_root / name / "detections.txt"
        if p.exists():
            return p
    return None


def quant(a, qs=(0.5, 0.9, 0.99)):
    if a.size == 0:
        return [float("nan")] * len(qs)
    return [float(np.quantile(a, q)) for q in qs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", required=True)
    ap.add_argument("--host-cache", required=True)
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--gt-dir", default="datasets/visem_tracking/mot_gt_c0")
    ap.add_argument("--board-script", default=str(HERE / "rknn_headraw_pipeline.py"))
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--fw", type=float, default=640.0)
    ap.add_argument("--fh", type=float, default=480.0)
    ap.add_argument("--gap", type=int, default=14)
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--low-match-iou", type=float, default=0.70)
    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--track-high", type=float, default=0.60)
    ap.add_argument("--new-track", type=float, default=0.60)
    ap.add_argument("--track-low", type=float, default=0.05)
    args = ap.parse_args()

    Tracker = load_board_tracker(Path(args.board_script))
    vids = [v for v in args.vids.split(",") if v]
    dump_root, host = Path(args.dump_root), Path(args.host_cache)

    print(f"{'vid':>4s} {'rows':>8s} {'frames':>7s} {'MOTA(rule)':>11s} "
          f"{'best scale':>11s} {'MOTA':>7s}")
    per_rule, rows_flat = {}, {}
    for v in vids:
        p = find_dump(dump_root, v)
        if p is None:
            print(f"{v:>4s}  no detections.txt under {dump_root}, skipped")
            continue
        pf, r = read_board(p, args.fw, args.fh)
        rows_flat[v] = r
        gt = parse_mot(HERE / args.gt_dir / f"{v}.txt")
        n = max(pf) if pf else 0
        gt = {f: b for f, b in gt.items() if 1 <= f <= n}

        rep = Replay(Tracker, args, pf, lambda st, fr: True,
                     args.fw, args.fh, emit_age=0)
        tr, _a, _b = rep.run(1, n, v, RULES5[v])
        m_rule = mota(gt, interpolate(tr, args.gap))["MOTA"]
        per_rule[v] = m_rule

        best, best_sc = None, None
        for sc in SWEEP:
            rep = Replay(Tracker, args, pf, lambda st, fr: True,
                         args.fw, args.fh, emit_age=0)
            tr, _a, _b = rep.run(1, n, v, sc)
            m = mota(gt, interpolate(tr, args.gap))["MOTA"]
            if best is None or m > best:
                best, best_sc = m, sc
        print(f"{v:>4s} {r.shape[0]:8d} {n:7d} {m_rule:11.4f} "
              f"{best_sc:11.2f} {best:7.4f}")

    print(f"\nboard INT8 with the kappa-rule scales: four-session mean MOTA "
          f"= {np.mean(list(per_rule.values())):.4f}")
    print("  host FP32 reference for E26: seed42 0.5704, four-seed mean 0.5290")

    print("\nINT8 vs FP32 detection statistics")
    for v in vids:
        r = rows_flat[v]
        npy = host / f"dets_{v}_{args.size}.npy"
        if not npy.exists():
            print(f"  {v}: host cache {npy.name} missing, skipped")
            continue
        chunks = [np.asarray(x, np.float64)
                  for x in np.load(npy, allow_pickle=True) if len(x)]
        h = np.concatenate(chunks) if chunks else np.zeros((0, 5))
        sb, sh = quant(r[:, 4]), quant(h[:, 4])
        wb, wh = quant(r[:, 2]), quant(h[:, 2])
        lv0 = r[r[:, 5] == 0] if r.shape[0] else r
        print(f"  {v}: n_board={r.shape[0]:6d} n_host={h.shape[0]:6d}")
        print(f"       score q50/q90/q99 board {sb[0]:.3f}/{sb[1]:.3f}/{sb[2]:.3f}"
              f"  host {sh[0]:.3f}/{sh[1]:.3f}/{sh[2]:.3f}"
              f"  ratio(q50) {sb[0] / max(sh[0], 1e-9):.3f}")
        print(f"       box w q50/q90/q99 board {wb[0]:.4f}/{wb[1]:.4f}/{wb[2]:.4f}"
              f"  host {wh[0]:.4f}/{wh[1]:.4f}/{wh[2]:.4f}"
              f"  ratio(q50) {wb[0] / max(wh[0], 1e-9):.3f}")
        if lv0.shape[0]:
            print(f"       level-0 mean w/h = {lv0[:, 2].mean():.4f} /"
                  f" {lv0[:, 3].mean():.4f}  (n={lv0.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
