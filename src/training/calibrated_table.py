#!/usr/bin/env python3
"""The table the paper needs: EVERY arm **plus our per-session calibration**.

The user's requirement (2026-09-22): the model-side change must be evaluated
together with the paper's per-session scale method, and must beat
"baseline + the same method" -- otherwise the two contributions look like
alternatives rather than a combination.

The scale used here is the label-free one from the paper: the kappa rule
    s = kappa * sqrt(f_mean * lapvar),  kappa = 3.066
shrunk by lambda = 0.5 (the deployed operating point), snapped to the scan grid.
For these four sessions that gives 0.95 / 1.05 / 1.20 / 1.05 -- and because the
rule reads *image* cues and one rig constant, the same four numbers apply to
every arm, which is exactly the deployment situation.

    python3 calibrated_table.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adaptive_sched_eval import Replay, load_board_tracker      # noqa: E402
from score_det_dump import interpolate, mota, parse_mot          # noqa: E402

# Per-session scale settings, for 82/60/54/52.
#   ORACLE  : each session at its measured optimum (what the delivered headline
#             0.66 uses) -- the ceiling, and the anchor for sanity checks
#   RULE    : the paper's label-free rule s = kappa*sqrt(f_mean*lapvar) with
#             kappa = 3.066 (out-of-sample fit), i.e. lambda = 1 (0.95/1.10/1.40/1.10)
#   RULE5   : the same rule shrunk by lambda = 0.5, the "no harm on trained
#             sessions" variant (0.95/1.05/1.20/1.05) -- deliberately conservative
ORACLE_SCALES = {"82": 0.90, "60": 1.40, "54": 1.50, "52": 0.95}
RULE_SCALES = {"82": 0.95, "60": 1.10, "54": 1.40, "52": 1.10}
RULE5_SCALES = {"82": 0.95, "60": 1.05, "54": 1.20, "52": 1.05}
# The four-cue rule the board actually shipped with
# (--box-scale = kappa*(f_mean*f_peak*hf_ratio*lapvar)**0.25, kappa = 15.735).
PIPELINE_SCALES = {"82": 0.80, "60": 1.10, "54": 1.25, "52": 0.95}
VAL = ["82", "60", "54", "52"]

# (label, cache dir, marker) -- caches produced by the arms' evaluation chains
ARMS = [
    ("baseline last.pt", "/root/autodl-tmp/rej_stock_yolov8n_448_last_val"),
    ("baseline best.pt", "/root/autodl-tmp/rej_stock_yolov8n_448_best_val"),
    ("E22 EMA@P3 last", "/root/autodl-tmp/res_E22"),
    ("E22 EMA@P3 best", "/root/autodl-tmp/best_v8n_ema448_val"),
    ("E12c self-cal last", "/root/autodl-tmp/res_e12c"),
    ("E12c self-cal best", "/root/autodl-tmp/best_gainc_val"),
    ("E12c self-cal seed7", "/root/autodl-tmp/res_E12cs7"),
    # E25's own cache (res_E25) was written at 15:11 by an overlapping earlier
    # launch and is STALE with respect to its 16:35 model, so it is re-dumped
    # into rej_E25_val before judging.
    ("E25 SE last", "/root/autodl-tmp/rej_E25_val"),
    ("E26 GatedCtx last", "/root/autodl-tmp/res_E26"),
    ("E20 subpix last", "/root/autodl-tmp/res_E20"),
    ("E21 tap last", "/root/autodl-tmp/res_E21"),
    # --- queued after E26/E25 (run_deployable_chain.sh + run_e28_chain.sh) ---
    # each entry is skipped automatically while its cache does not exist yet
    ("E27 subpix+SE last", "/root/autodl-tmp/res_E27"),
    ("E28 GatedCtx+selfcal last", "/root/autodl-tmp/res_E28"),
    ("E26 GatedCtx seed7", "/root/autodl-tmp/res_E26s7"),
    ("E29 GatedCtx P3+P4 last", "/root/autodl-tmp/res_E29"),
    ("E30 GatedCtx P5 last", "/root/autodl-tmp/res_E30"),
    ("E31 GatedCtx+half gain last", "/root/autodl-tmp/res_E31"),
    # --- seed robustness of the headline module (E26: seed42 +0.1056, seed7
    # +0.0557 -- a 0.050 spread against a +-0.017 baseline seed band) ---
    ("E26 GatedCtx seed123", "/root/autodl-tmp/res_E26s123"),
    ("E26 GatedCtx seed2024", "/root/autodl-tmp/res_E26s2024"),
    ("E32 GatedCtx in cv2[0] last", "/root/autodl-tmp/res_E32"),
    ("E31 GatedCtx+half gain seed7", "/root/autodl-tmp/res_E31s7"),
    # E33: the delivery module trained 60 epochs instead of 30 (E26's four seeds
    # spread 0.041..0.106 on the main table, so convergence was worth testing)
    ("E33 GatedCtx 60ep last", "/root/autodl-tmp/res_E33"),
    # E34: the delivery module with the two dead detection branches removed
    # (measured: 100% of surviving detections are level 0 in INT8 and FP32)
    ("E34 P3-only head last", "/root/autodl-tmp/res_E34"),
    ("E37 per-anchor gain last", "/root/autodl-tmp/res_E37"),
    ("E37b per-anchor gain v2 last", "/root/autodl-tmp/res_E37b"),
    ("E38 cue gain learned last", "/root/autodl-tmp/res_E38"),
    ("E33 GatedCtx 60ep last", "/root/autodl-tmp/res_E33"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="datasets/visem_tracking/mot_gt_c0")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--fw", type=float, default=640.0)
    ap.add_argument("--fh", type=float, default=480.0)
    ap.add_argument("--gap", type=int, default=14)
    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--track-high", type=float, default=0.60)
    ap.add_argument("--new-track", type=float, default=0.60)
    ap.add_argument("--track-low", type=float, default=0.05)
    # the DELIVERED k=1 work point is match_iou 0.30 (k>=2 switches to 0.20);
    # using 0.20 here silently cost the baseline ~0.01-0.02
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--low-match-iou", type=float, default=0.70)
    # Detection stride: 1 = the paper's main table; 2/3/4 reproduce the deployed
    # schedule (the board curve 13.2/28.7/40.4/50.6 FPS), so the same table can
    # be printed at the delivered frame rate.  Everything else is untouched.
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    Tracker = load_board_tracker(HERE / "rknn_headraw_pipeline.py")
    det_cache, gt_cache = {}, {}
    for label, cache in ARMS:
        cache = Path(cache)
        if not all((cache / f"dets_{v}_{args.size}.npy").exists() for v in VAL):
            continue
        det_cache[label] = {}
        gt_cache[label] = {}
        for v in VAL:
            dets = list(np.load(cache / f"dets_{v}_{args.size}.npy",
                                allow_pickle=True))
            gt = parse_mot(HERE / args.gt_dir / f"{v}.txt")
            n = len(dets)
            det_cache[label][v] = dets
            gt_cache[label][v] = {f: b for f, b in gt.items() if 1 <= f <= n}

    def run_at(label, scales):
        per = []
        for v in VAL:
            dets = det_cache[label][v]
            # Replay wants (level, score, (cx, cy, w, h)) in PIXELS: it feeds
            # them to the tracker unchanged and divides by fw/fh only when it
            # stores tracks.  The dumps are normalised (a first version passed
            # them through untouched -- every session then scored MOTA -0.99
            # with TP = 0 because the tracker held ~0-sized boxes).
            per_frame = {f: [(0, float(r[4]), (float(r[0]) * args.fw,
                                               float(r[1]) * args.fh,
                                               float(r[2]) * args.fw,
                                               float(r[3]) * args.fh))
                             for r in rows]
                         for f, rows in enumerate(dets, start=1) if len(rows)}
            if args.stride <= 1:
                decision = (lambda st, fr: True)
            else:
                decision = (lambda st, fr, k=args.stride: (fr - 1) % k == 0)
            rep = Replay(Tracker, args, per_frame,
                         decision, args.fw, args.fh, emit_age=0)
            tr, _a, _b = rep.run(1, len(dets), v, scales[v])
            per.append(mota(gt_cache[label][v],
                            interpolate(tr, args.gap))["MOTA"])
        return per

    if args.stride > 1:
        board_fps = {2: 28.7, 3: 40.4, 4: 50.6}.get(args.stride)
        print("\n*** DETECTION STRIDE k = %d ***  (board: %s FPS)"
              % (args.stride, "%.1f" % board_fps if board_fps else "n/a"))
    for tag, scales in (("ORACLE per-session scale (delivered headline protocol)",
                         ORACLE_SCALES),
                        ("PIPELINE label-free 4-cue rule (what the board ships)",
                         PIPELINE_SCALES),
                        ("LABEL-FREE rule, lambda=1.0 (kappa=3.066)",
                         RULE_SCALES),
                        ("LABEL-FREE rule, lambda=0.5 (no-harm variant)",
                         RULE5_SCALES),
                        ("NO calibration (s=1)", {v: 1.0 for v in VAL})):
        print(f"\n### {tag}   scales {[scales[v] for v in VAL]}")
        print(f"{'arm':>22s} " + " ".join(f"{v:>7s}" for v in VAL) +
              f" {'mean':>8s}  {'vs baseline(last)':>17s}")
        base = None
        for label in det_cache:
            per = run_at(label, scales)
            m = float(np.mean(per))
            if label.startswith("baseline last"):
                base = m
            d = "" if base is None else f"{m - base:+17.4f}"
            print(f"{label:>22s} " + " ".join(f"{x:7.4f}" for x in per) +
                  f" {m:8.4f}  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
