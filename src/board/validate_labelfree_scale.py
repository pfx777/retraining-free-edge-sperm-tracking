#!/usr/bin/env python3
"""Leave-one-session-out test of a label-free box-scale predictor.

The claim under test
--------------------
Object size and background texture come from the same optics, so an image
texture statistic measured with no labels may predict the per-session box-scale
correction that currently needs a small annotated slice.

Measured cue values (mean spatial frequency of the frame) and the measured
optimal scales line up on all four sessions:

    video   GT med px   optimal s   f_mean
    82           13.0        0.85   0.0509
    52           16.3        0.95   0.0594
    60           19.2        1.30   0.0641
    54           20.6        1.30   0.0756

-- and the decisive pair (54 needs 1.30, 82 needs 0.85) is separated by 39% on
that cue, which is what killed the "impossible" verdict.  With four sessions
though, a monotone ordering has a 1/24 chance of being luck, so this script does
the honest test: fit the cue -> scale map on three sessions, predict the fourth,
and score the MOTA that predictor would have delivered.

No inference and no GPU: the per-(model, video, scale) MOTA values already exist
in the pvscale sweep CSVs, so a prediction just snaps to the nearest grid point
and looks the number up.  Each of the four trained architectures gives an
independent test, because the cue depends only on the session while the optimal
scale also depends on the model.

Reference points reported next to it:
  * no calibration (scale 1.00)
  * the oracle over the grid (the upper bound the annotated protocol achieves)

    python validate_labelfree_scale.py --self-test
    python validate_labelfree_scale.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/code")
TRACK_ROOT = ROOT / "runs" / "visem_tracking"

# Cue values re-measured on a common budget (60 frames per session) by
# analyze_session_scale_cues.py, including the two sessions 29/36 that the
# cross-session model was retrained WITHOUT, so the leave-one-out test has six
# points instead of four.
#
# Two cues, because the failure of the single-cue version has a clear mechanism:
#   f_mean (mean spatial frequency) tracks how large the objects LOOK -> predicts
#       the annotated size, which is what the scalar is trying to match;
#   lapvar (Laplacian energy) tracks how SHARP the image is -> predicts how much
#       the model's own box output is inflated by the glow/halo.  Video 29 is the
#       extreme case: heavily out of focus (f_peak 0.0055 against 0.0136-0.0173
#       elsewhere, lapvar 1.1 against 1.8-3.1), the model follows its large halo
#       and outputs ~20 px for 14 px objects, so its optimal scale is 0.70 while
#       the single-cue model predicted 1.00.
CUES = {"36": 0.0498, "82": 0.0509, "52": 0.0594, "29": 0.0605,
        "60": 0.0641, "54": 0.0756}
CUES2 = {"82": 1.8, "60": 2.0, "54": 2.7, "52": 2.2, "29": 1.1, "36": 3.1}
# The two further texture cues analyze_session_scale_cues.py already measures.  They
# were left out of the earlier predictors only because the 3-parameter fit could not
# afford them; a geometric mean costs no extra parameter, so they can be added
# without trading against the fit's degrees of freedom.
#
#   f_peak   -- radius of the peak of the radial power spectrum
#   hf_ratio -- share of spectral power above the median radius
CUES3 = {"82": 0.0166, "60": 0.0173, "54": 0.0151, "52": 0.0136, "29": 0.0055, "36": 0.0138}
CUES4 = {"82": 0.0045, "60": 0.0106, "54": 0.0128, "52": 0.0072, "29": 0.0060, "36": 0.0060}
MODELS = {
    "dyhead30_448": "pvscale_sweep_ablation_dyhead30.csv",
    "nop2_448": "pvscale_sweep_ablation_nop2448.csv",
    "loctap_448": "pvscale_sweep_ablation_loctap448.csv",
    "ubwa_448": "pvscale_sweep_ablation_ubwa448.csv",
    # the remaining architectures that already have a per-video scale sweep: every
    # one of them is a different detector, so each is an independent test of the
    # same cue -> scale rule (the rule depends on the SESSION, the optimum also on
    # the model).  lkctx30 is absent because its only sweep is at imgsz 640 and
    # covers a single session, which cannot carry a leave-one-out test.
    "loctapc_448": "pvscale_sweep_ablation_loctapc448.csv",
    "loctapp_448": "pvscale_sweep_ablation_loctapp448.csv",
    "nohead_448": "pvscale_sweep_ablation_nohead448.csv",
    "p2_448": "pvscale_sweep_ablation_p2448.csv",
    "ubwa2_448": "pvscale_sweep_ablation_ubwa2448.csv",
    "regmdfm_448": "pvscale_sweep_ablation_regmdfm448.csv",
    # trained WITHOUT 29/36, and swept over six sessions -- the honest test
    "loctapnb_448_6sess": "pvscale_sweep_ablation_loctapnb448.csv",
}


def fit_log_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares ``log y = a + b log x``."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    den = sum((x - mx) ** 2 for x in lx)
    b = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / den if den > 0 else 0.0
    return my - b * mx, b


def nearest_scale(scales: list[float], s: float) -> float:
    return min(scales, key=lambda c: abs(math.log(c) - math.log(s)))


def geometric_mix(vid: str, mode: str, drop: set[str]) -> float:
    """The cue combination the rule feeds into its single fitted constant.

    Equal weights in log space, i.e. an unweighted geometric mean.  That is the
    only choice that adds no free parameter: every alternative (weights, cue
    selection by performance) would have to be fitted on the same handful of
    sessions that the rule is then scored on.
    """
    named = [("fmean", CUES), ("fpeak", CUES3), ("hf", CUES4), ("lapvar", CUES2)]
    if mode == "sqrt":
        named = [("fmean", CUES), ("lapvar", CUES2)]
    elif mode == "all3":
        named = [x for x in named if x[0] != "fpeak"]
    used = [d[vid] for name, d in named if name not in drop]
    return math.exp(sum(math.log(u) for u in used) / len(used))


def apply_guard(pred_raw: float, shrink: float, gate: float) -> float:
    """Turn the raw prediction into the scale actually shipped, honestly bounded.

    Two guards, both applied in log space and both pre-registered here before any
    of them was measured:

    * ``shrink`` pulls the prediction toward 1.00 by a factor in [0, 1].  A
      correction is only worth making if the predictor is more often right than
      wrong, so pulling toward the no-correction default trades upside for a
      smaller worst case.
    * ``gate`` leaves the session uncorrected when ``|log s| < gate``.  This is
      the same idea with a dead zone: a correction smaller than the predictor's
      own error is not a correction, it is noise with a sign.

    Both were motivated by the same observation -- the folds the rule LOST were
    the ones where it asked for a ~5% change, while the folds it won asked for
    20-25% -- and neither is fitted on the held-out session.
    """
    s = math.exp(shrink * math.log(pred_raw))
    if gate > 0.0 and abs(math.log(s)) < gate:
        return 1.0
    return s


def fit_two_feature(x1: list[float], x2: list[float], y: list[float]) -> tuple[float, float, float]:
    """Least squares for ``log y = a + b x1 + c x2`` (x1, x2 already logged).

    Three parameters on five training sessions is not much, but the point of the
    second cue is mechanistic rather than statistical: the single-cue version's
    failures are exactly the sessions whose images are out of focus, so a
    sharpness term has a reason to be there.
    """
    A = np.column_stack([np.ones(len(x1)), np.asarray(x1, float), np.asarray(x2, float)])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def knn_predict(cue1: list[float], cue2: list[float], ys: list[float],
                q1: float, q2: float, k: int) -> float:
    """Distance-weighted k-NN in standardised log-cue space.

    Standardisation uses the training sessions only.  The point of offering this
    next to the regression is that it CANNOT extrapolate: a session that looks
    unlike anything in the training set still receives a scale that some real
    session justified.  That matters for deployment -- shipping a predictor whose
    regression occasionally emits 2.03 (as the two-feature fit did on video 54)
    is worse than shipping one that is merely conservative.
    """
    l1 = [math.log(v) for v in cue1]
    l2 = [math.log(v) for v in cue2]
    m1 = sum(l1) / len(l1)
    m2 = sum(l2) / len(l2)
    s1 = math.sqrt(sum((x - m1) ** 2 for x in l1) / len(l1)) or 1.0
    s2 = math.sqrt(sum((x - m2) ** 2 for x in l2) / len(l2)) or 1.0
    qa, qb = (math.log(q1) - m1) / s1, (math.log(q2) - m2) / s2
    dist = [math.hypot((x - m1) / s1 - qa, (y - m2) / s2 - qb) for x, y in zip(l1, l2)]
    order = sorted(range(len(dist)), key=lambda i: dist[i])[:k]
    if len(order) == 1 or dist[order[0]] < 1e-12:
        return ys[order[0]]
    w = [1.0 / max(dist[i], 1e-9) for i in order]
    tot = sum(w)
    return math.exp(sum(wi * math.log(ys[i]) for wi, i in zip(w, order)) / tot)


def halo_predict(cue1: list[float], cue2: list[float], ys: list[float],
                 q1: float, q2: float) -> float:
    """Two-parameter fit of the halo-inflation model, ``s = c / (1 + k r)``.

    Derivation.  Let ``a`` be the apparent object size, ``sigma`` the defocus
    blur and ``g = c a`` the annotated size (the annotator follows the visible
    halo).  The detector inherits the halo, so it predicts ``d = a + k' sigma``.
    The scale that aligns prediction with annotation is therefore

        s = g / d = c / (1 + k' sigma / a).

    ``f_mean`` falls as the objects get larger (spatial-frequency content scales
    as 1/a) and ``lapvar`` rises as the image sharpens (roughly 1/sigma^2), so
    ``r = f_mean / sqrt(lapvar)`` is the relative blur, and the whole model has
    exactly two free numbers: ``c`` and ``k``.  For a fixed ``k`` the optimal
    ``c`` is the geometric mean of ``s_i (1 + k r_i)``, so the fit is a 1-D
    search -- no risk of the degenerate 3-parameter behaviour seen above.
    """
    rs = [x / math.sqrt(y) for x, y in zip(cue1, cue2)]
    qr = q1 / math.sqrt(q2)

    def cost(k: float) -> float:
        logc = sum(math.log(s) + math.log1p(k * r) for s, r in zip(ys, rs)) / len(ys)
        return sum((math.log(s) - (logc - math.log1p(k * r))) ** 2 for s, r in zip(ys, rs))

    lo, hi = 1e-3, 1e3
    for _ in range(200):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if cost(m1) < cost(m2):
            hi = m2
        else:
            lo = m1
    k = 0.5 * (lo + hi)
    logc = sum(math.log(s) + math.log1p(k * r) for s, r in zip(ys, rs)) / len(ys)
    return math.exp(logc - math.log1p(k * qr))


def self_test() -> int:
    a, b = fit_log_line([1.0, 2.0, 4.0], [2.0, 4.0, 8.0])
    assert abs(b - 1.0) < 1e-9 and abs(a - math.log(2)) < 1e-9, (a, b)
    assert abs((a + b * math.log(8.0)) - math.log(16.0)) < 1e-9, (a, b)
    assert nearest_scale([0.8, 0.9, 1.0], 0.87) == 0.9
    assert nearest_scale([0.8, 0.9, 1.0], 5.0) == 1.0
    # a flat cue carries no information and must fall back to a zero slope
    a2, b2 = fit_log_line([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    assert b2 == 0.0, (a2, b2)
    print("self-test OK: log-linear fit recovers an exact power law, and prediction "
          "snapping picks the grid point nearest in log space")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--vids", default="82,60,54,52,29,36")
    ap.add_argument("--fit-rig", action="store_true",
                    help="print the ONE constant a rig needs: fit kappa on every annotated "
                         "session and freeze it (in-sample, hence optimistic by construction)")
    ap.add_argument("--shrink", type=float, default=1.0,
                    help="pull the predicted log-scale toward 0 (no correction) by this factor")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="leave a session uncorrected when |log s_pred| is below this dead zone")
    ap.add_argument("--gate-nested", action="store_true",
                    help="set the dead zone to --gate-k x the predictor's OWN residual RMS on the "
                         "training folds only -- no number is chosen by looking at the held-out "
                         "session, which is what makes a guard legitimate rather than fitted")
    ap.add_argument("--gate-k", type=float, default=1.0,
                    help="multiplier on that nested residual scale (1 = 'do not act on a "
                         "correction smaller than your own typical error')")
    ap.add_argument("--drop", default="",
                    help="comma-separated cues to drop from the all4 geometric mean, any of "
                         "fmean,fpeak,hf,lapvar -- the falsification test for whether the "
                         "result rests on one column")
    ap.add_argument("--cues-file", default=None,
                    help="replace the built-in cue values with those measured on another frame "
                         "sample (analyze_session_scale_cues.py --cues-out).  Re-running the "
                         "whole validation on several samples is the test of whether the result "
                         "survives a change of nothing but the frames it was measured on.")
    ap.add_argument("--cue", choices=("fmean", "lapvar", "both", "knn1", "knn2", "ratio", "prod",
                                      "halo", "sqrt", "all4", "all3"),
                    default="both",
                    help="fmean = texture only (the original single cue); lapvar = sharpness "
                         "only; both = log-linear fit on texture and sharpness; "
                         "knn1/knn2 = nearest / distance-weighted two neighbours on both cues; "
                         "ratio/prod = the two-parameter fit on the f_mean/lapvar ratio or product")
    return ap.parse_args()


def load_table(path: Path, imgsz: int) -> tuple[dict[str, dict[float, float]], list[float]]:
    """``{video: {scale: MOTA}}`` plus the scale grid, for one sweep CSV."""
    table: dict[str, dict[float, float]] = {}
    scales: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("video") == "MEAN" or int(r.get("imgsz", 0)) != imgsz:
                continue
            vals = {}
            for k, v in r.items():
                if k.startswith("mota@") and v not in ("", "nan"):
                    try:
                        vals[float(k[5:])] = float(v)
                    except ValueError:
                        pass
            if vals:
                table[r["video"]] = vals
                scales = sorted(vals)
    return table, scales


def rig_fit(model: str, table: dict[str, dict[float, float]], scales: list[float],
            mode: str, drop: set[str]) -> None:
    """The per-rig calibration step: four numbers in, ONE constant out.

    A new rig (or a new detector on the same rig) follows this once:
      1. pick the annotated sessions that already exist for the rig;
      2. run the box-scale sweep on them and take each session's argmax scale;
      3. fit ``kappa`` as the geometric mean of ``s_opt / sqrt(f_mean * lapvar)``;
      4. freeze it.  Every later session is then calibrated with zero labels.

    The number printed here is IN-SAMPLE -- kappa has seen these sessions -- so it
    is an upper bound.  The honest number is the leave-one-out table below.
    """
    vids = [v for v in sorted(CUES) if v in table]
    if len(vids) < 2:
        print(f"  {model}: fewer than two sessions with both cues and sweeps")
        return
    s_opt = {v: max(table[v], key=lambda s: table[v][s]) for v in vids}
    mix = {v: geometric_mix(v, mode, drop) for v in vids}
    kappa = math.exp(sum(math.log(s_opt[v]) - math.log(mix[v]) for v in vids) / len(vids))
    print(f"\n  {model}:  kappa = {kappa:.3f}   (fitted on {len(vids)} annotated sessions)")
    print(f"    {'video':<7}{'cue':>9}{'s_opt':>8}{'s_pred':>8}{'snap':>7}"
          f"{'MOTA@pred':>11}{'MOTA@oracle':>13}")
    tot_p = tot_o = 0.0
    for v in vids:
        raw = kappa * mix[v]
        snap = nearest_scale(scales, raw)
        m_p, m_o = table[v].get(snap, float("nan")), max(table[v].values())
        tot_p += m_p
        tot_o += m_o
        print(f"    {v:<7}{mix[v]:>9.4f}{s_opt[v]:>8.2f}{raw:>8.2f}"
              f"{snap:>7.2f}{m_p:>11.3f}{m_o:>13.3f}")
    print(f"    {'mean':<7}{'':>9}{'':>8}{'':>8}{'':>7}{tot_p / len(vids):>11.3f}"
          f"{tot_o / len(vids):>13.3f}   <-- IN-SAMPLE, optimistic")
    print(f"    freeze kappa = {kappa:.3f}; every later session on this rig needs zero labels.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()

    if args.cues_file:
        for vid, c in json.loads(Path(args.cues_file).read_text()).items():
            CUES[vid], CUES2[vid] = c["f_mean"], c["lapvar"]
            CUES3[vid], CUES4[vid] = c["f_peak"], c["hf_ratio"]
        print(f"cues from {args.cues_file}")

    vids = [v.strip() for v in args.vids.split(",") if v.strip()]
    rows_any = False
    summary = []
    for model, fname in MODELS.items():
        path = TRACK_ROOT / fname
        if not path.exists():
            print(f"  {model}: missing {path.name}")
            continue
        table, scales = load_table(path, args.imgsz)
        if args.fit_rig:
            drop = {d.strip() for d in args.drop.split(",") if d.strip()}
            rig_fit(model, table, scales, args.cue, drop)
            continue
        if not table:
            print(f"  {model}: no imgsz={args.imgsz} rows in {path.name}")
            continue
        rows_any = True
        per_vid = []
        for held in vids:
            if held not in table or held not in CUES:
                continue
            train = [v for v in vids if v in table and v in CUES and v != held]
            if len(train) < 2:
                continue
            s_opt = {v: max(table[v], key=lambda s: table[v][s]) for v in train}
            if args.cue == "both":
                a, b, c = fit_two_feature(
                    [math.log(CUES[v]) for v in train],
                    [math.log(CUES2[v]) for v in train],
                    [math.log(s_opt[v]) for v in train],
                )
                pred_raw = math.exp(a + b * math.log(CUES[held]) + c * math.log(CUES2[held]))
            elif args.cue in ("knn1", "knn2"):
                pred_raw = knn_predict(
                    [CUES[v] for v in train], [CUES2[v] for v in train],
                    [s_opt[v] for v in train], CUES[held], CUES2[held],
                    1 if args.cue == "knn1" else 2,
                )
            elif args.cue in ("ratio", "prod"):
                mix = {v: (CUES[v] / CUES2[v] if args.cue == "ratio" else CUES[v] * CUES2[v])
                       for v in vids}
                a, b = fit_log_line([mix[v] for v in train], [s_opt[v] for v in train])
                pred_raw = math.exp(a + b * math.log(mix[held]))
            elif args.cue == "sqrt":
                mix = {v: CUES[v] * CUES2[v] for v in vids}
                a = sum(math.log(s_opt[v]) - 0.5 * math.log(mix[v]) for v in train) / len(train)
                pred_raw = math.exp(a + 0.5 * math.log(mix[held]))
                # the predictor's own error, measured on the training folds only
                sigma = math.sqrt(sum((math.log(s_opt[v]) - a - 0.5 * math.log(mix[v])) ** 2
                                      for v in train) / len(train))
            elif args.cue in ("all4", "all3"):
                # unweighted geometric mean of every measured cue (all3 = the same
                # minus f_peak, whose ordering is the one that disagrees with s_opt)
                drop = {d.strip() for d in args.drop.split(",") if d.strip()}
                mix = {v: geometric_mix(v, args.cue, drop) for v in vids}
                a = sum(math.log(s_opt[v]) - math.log(mix[v]) for v in train) / len(train)
                pred_raw = math.exp(a + math.log(mix[held]))
                sigma = math.sqrt(sum((math.log(s_opt[v]) - a - math.log(mix[v])) ** 2
                                      for v in train) / len(train))
            elif args.cue == "halo":
                pred_raw = halo_predict(
                    [CUES[v] for v in train], [CUES2[v] for v in train],
                    [s_opt[v] for v in train], CUES[held], CUES2[held],
                )
            else:
                src = CUES if args.cue == "fmean" else CUES2
                a, b = fit_log_line([src[v] for v in train], [s_opt[v] for v in train])
                pred_raw = math.exp(a + b * math.log(src[held]))
            gate = args.gate
            if args.gate_nested and args.cue in ("sqrt", "all4", "all3"):
                gate = args.gate_k * sigma
            pred = nearest_scale(scales, apply_guard(pred_raw, args.shrink, gate))
            m_pred = table[held].get(pred, float("nan"))
            m_none = table[held].get(1.0, float("nan"))
            m_oracle = max(table[held].values())
            s_oracle = max(table[held], key=lambda s: table[held][s])
            hit = abs(math.log(pred) - math.log(s_oracle)) < 1e-9
            per_vid.append((held, pred_raw, pred, m_none, m_pred, m_oracle, s_oracle, hit))
        if not per_vid:
            continue
        means = [sum(p[i] for p in per_vid) / len(per_vid) for i in (3, 4, 5)]
        summary.append((model, means, sum(p[7] for p in per_vid), len(per_vid)))
        print(f"\n  {model}")
        print(f"    {'held-out':<10}{'predicted s':>13}{'snapped':>9}{'oracle s':>10}"
              f"{'none (1.00)':>13}{'label-free':>12}{'oracle':>9}")
        for held, raw, snap, m_none, m_pred, m_or, s_or, hit in per_vid:
            mark = "  ok" if hit else ""
            print(f"    {held:<10}{raw:>13.2f}{snap:>9.2f}{s_or:>10.2f}"
                  f"{m_none:>13.3f}{m_pred:>12.3f}{m_or:>9.3f}{mark}")
        print(f"    {'mean':<10}{'':>13}{'':>9}{'':>10}{means[0]:>13.3f}{means[1]:>12.3f}"
              f"{means[2]:>9.3f}   ({summary[-1][2]}/{summary[-1][3]} folds land on the oracle scale)")

    if not rows_any:
        if args.fit_rig:
            return 0
        print("no sweep CSVs found; nothing to test")
        return 1
    print("\n" + "=" * 82)
    print("label-free predictor vs no calibration and vs the oracle (mean over held-out sessions)")
    print(f"  {'model':<16}{'none':>10}{'label-free':>13}{'oracle':>10}"
          f"{'captured':>11}{'@oracle':>9}")
    for model, (m_none, m_pred, m_or), hits, nfold in summary:
        span = m_or - m_none
        cap = (m_pred - m_none) / span if abs(span) > 1e-9 else float("nan")
        print(f"  {model:<16}{m_none:>10.3f}{m_pred:>13.3f}{m_or:>10.3f}{cap:>10.0%}"
              f"{hits:>6}/{nfold}")
    print("\n  'captured' = share of the annotated-calibration gain that the label-free")
    print("  predictor would have recovered without any labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
