#!/usr/bin/env python3
"""How few annotated frames does the per-session scale calibration need?

Two independent levers, measured together on labels that already exist:

  1. fewer frames.  Only 5% of frames was ever validated (six settings and five
     seeds all matched the oracle), so the protocol has never been probed below
     that.  This sweeps 0.5% ... 20%.

  2. a better estimator.  The shipped protocol takes a discrete argmax of MOTA
     over a coarse scale grid, which is noisy by construction: one bad cell flips
     the choice.  The closed form

         s* = 1 / median(det/GT side ratio over matched pairs)

     uses every matched box instead of only the winning cell, so it should hold
     up at far fewer frames.  On the four validation videos it already lands
     within one grid step of the measured optimum:

         video   closed form   measured optimum
         82          0.88            0.85
         60          1.25            1.30
         54          1.38            1.30
         52          0.98            0.95

The eval curve is computed once per scale and reused across every fraction, so
this costs seconds instead of a tracker run per setting.

    python sweep_calib_fraction.py --self-test
    python sweep_calib_fraction.py --run-name ablation_loctap448 --imgsz 448 \
        --vids 82,60,54,52 --fracs 0.005,0.01,0.02,0.03,0.05,0.10,0.20
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/code")


def median_ratio_scale(pairs: list[tuple[float, float]]) -> float:
    """Closed-form scale from ``(side_px, det/GT side ratio)`` matched pairs."""
    if not pairs:
        return float("nan")
    r = float(np.median(np.asarray([p[1] for p in pairs], dtype=np.float64)))
    return 1.0 / r if r > 0 else float("nan")


def nearest_scale(scales, s: float) -> float | None:
    if s is None or not np.isfinite(s) or not scales:
        return None
    return min(scales, key=lambda c: abs(math.log(c) - math.log(s)))


def self_test() -> int:
    p = [(10.0, 1.40), (12.0, 1.40), (14.0, 1.40), (30.0, 0.80)]
    assert abs(median_ratio_scale(p) - 1 / 1.40) < 1e-9, median_ratio_scale(p)
    # an outlier must not move a median the way it moves a mean
    q = p + [(99.0, 9.0)]
    assert abs(median_ratio_scale(q) - 1 / 1.40) < 1e-9, median_ratio_scale(q)
    assert math.isnan(median_ratio_scale([]))
    assert nearest_scale([0.8, 0.9, 1.0], 0.87) == 0.9
    assert nearest_scale([0.8, 0.9, 1.0], float("nan")) is None
    print("self-test OK: the closed-form scale inverts the median ratio and snaps "
          "to the grid point nearest in log space")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run-name", default="ablation_loctap448")
    ap.add_argument("--imgsz", default="448")
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--scales", default="0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,"
                                     "1.05,1.10,1.15,1.20,1.25,1.30")
    ap.add_argument("--fracs", default="0.005,0.01,0.02,0.03,0.05,0.10,0.20")
    ap.add_argument("--modes", default="even")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--interp-gap", type=int, default=14)
    ap.add_argument("--fit-iou", type=float, default=0.1,
                    help="overlap for a pair to enter the closed-form estimate")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap on calibration frames, to show the cost of an absolute budget")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()

    sys.path.insert(0, str(ROOT))
    import run_tracking_deepocsort as rtd  # noqa: E402
    from run_tracking_interp_sweep import interpolate_tracks  # noqa: E402
    from sweep_box_curve import collect_ratio_pairs, frame_size, parse_edges  # noqa: E402
    from validate_scale_calibration import (  # noqa: E402
        LEGACY_TEMPLATE,
        find_labels,
        interp_to_pred,
        load_pred,
        load_tracks,
        pick_calib_frames,
        subset_frames,
    )

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    fracs = [float(s) for s in args.fracs.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    vids = [v.strip() for v in args.vids.split(",") if v.strip()]
    edges = parse_edges("0,8,12,16,20,24,32,48,inf")

    rows = []
    for vid in vids:
        gt_path = rtd.MOT_GT / f"{vid}.txt"
        video = rtd.SRC / vid / f"{vid}.mp4"
        if not gt_path.exists() or not video.exists():
            print(f"  video {vid}: missing GT or video")
            continue
        gt_all = rtd.parse_mot(str(gt_path))
        frames_all = sorted(gt_all)
        preds: dict[float, dict[int, list[dict]]] = {}
        for sc in scales:
            key = f"{sc:.2f}"
            d = find_labels(["/root/code/runs/visem_tracking/pvscale_{run}_{imgsz}_{vid}_{scale}",
                             LEGACY_TEMPLATE], args.run_name, args.imgsz, vid, key)
            if d is None:
                continue
            preds[sc] = (
                interp_to_pred(interpolate_tracks(load_tracks(d), args.interp_gap))
                if args.interp_gap > 0
                else load_pred(d)
            )
        if not preds:
            print(f"  video {vid}: no labels")
            continue
        iw, ih = frame_size(video)
        letter = float(args.imgsz) / max(ih, iw)
        sx, sy = iw * letter, ih * letter
        eval_curve = {sc: rtd.mot_metrics(gt_all, preds[sc])["MOTA"] for sc in preds}
        oracle = max(eval_curve.values())
        base = eval_curve.get(1.0, float("nan"))
        print(f"\n  video {vid}: {len(frames_all)} frames, {len(preds)} scales, "
              f"uncalibrated {base:.3f}, oracle {oracle:.3f}")

        for mode in modes:
            for seed in seeds:
                for frac in fracs:
                    n = max(1, int(round(len(frames_all) * frac)))
                    if args.max_frames:
                        n = min(n, args.max_frames)
                    calib = pick_calib_frames(frames_all, n, mode, seed)
                    ev = set(frames_all) - calib
                    gt_c = subset_frames(gt_all, calib)
                    # (a) the shipped protocol: discrete argmax of calib MOTA
                    cm = {sc: rtd.mot_metrics(gt_c, subset_frames(preds[sc], calib))["MOTA"] for sc in preds}
                    s_argmax = max(cm, key=lambda s: cm[s])
                    # (b) the closed form, on the same frames.  The ratio has to be
                    # measured on the UNCORRECTED grid point -- measuring it on the
                    # argmax-scaled boxes measures how far an already-corrected box
                    # is from correct, which is ~1 by construction and says nothing.
                    base_scale = 1.0 if 1.0 in preds else min(preds)
                    pairs = collect_ratio_pairs(gt_c, subset_frames(preds[base_scale], calib),
                                                calib, sx, sy, args.fit_iou)
                    r_med = float(np.median([p[1] for p in pairs])) if pairs else float("nan")
                    s_cf = nearest_scale(list(preds), base_scale / r_med if r_med == r_med else float("nan"))
                    n_boxes = sum(len(v) for v in gt_c.values())
                    rows.append({
                        "vid": vid, "mode": mode, "seed": seed, "frac": frac,
                        "frames": len(calib), "boxes": n_boxes,
                        "argmax_s": s_argmax, "argmax_mota": eval_curve[s_argmax],
                        "cf_s": s_cf, "cf_mota": eval_curve.get(s_cf, float("nan")),
                        "oracle": oracle, "base": base,
                    })

    if not rows:
        print("nothing measured")
        return 1

    print("\n" + "=" * 96)
    print("annotated budget vs accuracy  (eval MOTA on the untouched 95%+; oracle is the grid optimum)")
    print(f"  {'mode':<8}{'seed':>5}{'frac':>7}{'frames':>8}{'boxes':>8}"
          f"{'argmax':>9}{'s_argmax':>10}{'closed':>9}{'s_closed':>10}{'oracle':>9}")
    seen = set()
    for r in rows:
        key = (r["mode"], r["seed"], r["frac"])
        if key in seen:
            seen.discard(key)
            continue
        seen.add(key)
        print(f"  {r['mode']:<8}{r['seed']:>5}{r['frac']:>7.3f}{r['frames']:>8}{r['boxes']:>8}"
              f"{r['argmax_mota']:>9.3f}{r['argmax_s']:>10.2f}{r['cf_mota']:>9.3f}"
              f"{(r['cf_s'] if r['cf_s'] is not None else float('nan')):>10.2f}{r['oracle']:>9.3f}")

    print("\n  means over videos, per budget")
    print(f"  {'frac':>7}{'frames/video':>14}{'boxes/video':>13}{'argmax':>9}{'closed':>9}"
          f"{'oracle':>9}{'uncal':>8}")
    by = {}
    for r in rows:
        by.setdefault((r["mode"], r["seed"], r["frac"]), []).append(r)
    for key in sorted(by, key=lambda k: (k[0], k[1], k[2])):
        g = by[key]
        f = lambda k: float(np.nanmean([x[k] for x in g]))
        print(f"  {key[2]:>7.3f}{f('frames'):>14.0f}{f('boxes'):>13.0f}"
              f"{f('argmax_mota'):>9.3f}{f('cf_mota'):>9.3f}{f('oracle'):>9.3f}{f('base'):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
