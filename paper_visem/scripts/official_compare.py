#!/usr/bin/env python3
"""Official VISEM-Tracking checkpoints vs our detector, under one protocol.

Same frames, same tracker (the one shipped on the board), same MOTA
(class-0 GT, IoU>=0.5, id switches charged, gaps<=14 interpolated), same
per-session output box scale.  Two questions:

  1. how does the official baseline (YOLOv5n / YOLOv5x at img 640, 3-class
     checkpoints filtered to class 0) compare with our YOLOv8n-448?
  2. does the *label-free calibration* transfer to a detector we did not train?
     kappa is refitted for each model on the 16 training sessions from the image
     clues (c_v = s_pred_v / 3.066, derived from logs/kappa_outfit.log), then the
     same rule (fixed lambda=0.5 and the q90-driven dynamic lambda) is applied on
     the 4 validation sessions.

    python official_compare.py \
        --val-dets data/official_dets/dets_val_yolov5x \
        --labels yolov5x ...
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT.parent
sys.path.insert(0, str(CODE))

from adaptive_sched_eval import Replay, load_board_tracker          # noqa: E402
from board_trk_replay import scale_tracks                            # noqa: E402
from score_det_dump import interpolate, mota, parse_mot              # noqa: E402

FW, FH = 640.0, 480.0
VAL = ["82", "60", "54", "52"]
TRAIN = ["11", "12", "13", "14", "15", "19", "21", "22", "23", "24",
         "29", "30", "35", "36", "38", "47"]
KAPPA_OURS = 3.066
GRID = np.round(np.arange(0.60, 1.61, 0.05), 2)
GRID_COARSE = np.round(np.arange(0.70, 1.71, 0.10), 2)
RULE_VAL = {"82": 0.95, "60": 1.10, "54": 1.40, "52": 1.10}   # kappa rule, lambda=1


class Args:
    track_high = 0.60
    new_track = 0.60
    track_low = 0.05
    match_iou = 0.30
    low_match_iou = 0.70
    max_age = 60


def load_txt_dump(path: Path):
    """official_<vid>.txt rows: frame cx cy w h score (normalised)."""
    per = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        fr = int(float(p[0]))
        cx, cy, w, h, s = (float(p[1]) * FW, float(p[2]) * FH,
                           float(p[3]) * FW, float(p[4]) * FH, float(p[5]))
        per.setdefault(fr, []).append((0, s, (cx, cy, w, h)))
    return per, (max(per) if per else 0)


def load_npy_cache(cache: Path, v: str, size: int = 448):
    dets = list(np.load(cache / f"dets_{v}_{size}.npy", allow_pickle=True))
    per = {}
    for f, rows in enumerate(dets, start=1):
        if len(rows):
            per[f] = [(0, float(r[4]), (float(r[0]) * FW, float(r[1]) * FH,
                                        float(r[2]) * FW, float(r[3]) * FH)) for r in rows]
    return per, len(dets)


def one_pass(tracker_cls, per, hi, v):
    rep = Replay(tracker_cls, Args, per, lambda st, fr: True, FW, FH, emit_age=0)
    tr, _a, _b = rep.run(1, hi, v, 1.0)      # track on unscaled boxes; scale later
    return tr


def interp_scaled(tr, scale, gap=14):
    """Interpolate once, then scale (the two operations commute per box)."""
    interp = interpolate(tr, gap)
    return {f: [(tid, (b[0], b[1], b[2] * scale, b[3] * scale)) for tid, b in rows]
            for f, rows in interp.items()}


def mota_at(tr, gt, scale, gap=14):
    return mota(gt, interp_scaled(tr, scale, gap))["MOTA"]


def best_scale(tr, gt, gap=14):
    best, best_s = None, None
    for s in GRID:
        m = mota_at(tr, gt, float(s), gap)
        if best is None or m > best:
            best, best_s = m, float(s)
    return best_s, best


def _train_session(job):
    """Worker: one training session -> its measured optimum scale."""
    v, dets_dir, gt_train, clue_v = job
    tracker_cls = load_board_tracker(CODE / "rknn_headraw_pipeline.py")
    p = Path(dets_dir) / f"official_{v}.txt"
    if not p.exists() or clue_v is None:
        return None
    per, hi = load_txt_dump(p)
    gt = parse_mot(Path(gt_train) / f"{v}.txt")
    gt = {f: b for f, b in gt.items() if 1 <= f <= hi}
    tr = one_pass(tracker_cls, per, hi, v)
    interp = interpolate(tr, 14)
    best, best_s = None, None
    for s in GRID_COARSE:
        m = mota(gt, {f: [(t, (b[0], b[1], b[2] * float(s), b[3] * float(s)))
                          for t, b in rows] for f, rows in interp.items()})["MOTA"]
        if best is None or m > best:
            best, best_s = m, float(s)
    m_none = mota(gt, interp)["MOTA"]
    return v, best_s, best, m_none, q90(per)


def q90(per):
    scores = [s for rows in per.values() for _lv, s, _b in rows]
    return float(np.quantile(scores, 0.9)) if scores else float("nan")


def parse_kappa_log(path: Path) -> dict[str, float]:
    """s_pred per session (kappa rule, lambda=1) -> clue term c_v = s_pred / kappa."""
    clue = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(\d{2})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s", line)
        if m:
            vid, s_opt, s_pred = m.group(1), float(m.group(2)), float(m.group(3))
            clue[vid] = {"s_opt_ours": s_opt, "s_pred": s_pred, "clue": s_pred / KAPPA_OURS}
    return clue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dets", required=True, help="dir with official_<vid>.txt for the 4 val sessions")
    ap.add_argument("--val-npy", default=None,
                    help="alternative: dir with dets_<vid>_448.npy (this project's host FP32 caches)")
    ap.add_argument("--train-dets", default=None, help="dir with official_<vid>.txt for the 16 train sessions")
    ap.add_argument("--label", default="official")
    ap.add_argument("--gt-val", default=str(CODE / "datasets" / "visem_tracking" / "mot_gt_c0"))
    ap.add_argument("--gt-train", default=str(ROOT / "data" / "gt_train"))
    ap.add_argument("--kappa-log", default=str(ROOT / "data" / "kappa_outfit.log"))
    ap.add_argument("--out", default=str(ROOT / "data" / "official_compare.json"))
    ap.add_argument("--jobs", type=int, default=8, help="worker processes for the training-session kappa fit")
    args = ap.parse_args()

    tracker_cls = load_board_tracker(CODE / "rknn_headraw_pipeline.py")
    clue = parse_kappa_log(Path(args.kappa_log)) if Path(args.kappa_log).exists() else {}
    for v, s in RULE_VAL.items():
        clue.setdefault(v, {})["clue"] = s / KAPPA_OURS
        clue[v]["s_pred"] = s

    result = {"label": args.label, "kappa_ours": KAPPA_OURS, "clue": {k: v.get("clue") for k, v in clue.items()}}

    # ---------- fit kappa for this detector on the 16 training sessions ----------
    fitted = None
    if args.train_dets:
        jobs = [(v, args.train_dets, args.gt_train, clue.get(v, {}).get("clue")) for v in TRAIN]
        with mp.Pool(processes=args.jobs) as pool:
            outs = [o for o in pool.map(_train_session, jobs) if o]
        ratios, per_train = [], {}
        for v, s_opt, m_opt, m_none, q in sorted(outs):
            c = clue[v]["clue"]
            ratios.append(s_opt / c)
            per_train[v] = {"s_opt": s_opt, "mota_opt": m_opt, "mota_none": m_none,
                            "clue": c, "q90": q}
            print(f"  train {v}: s_opt {s_opt:.2f}  clue {c:.3f}  ratio {s_opt/c:.3f}  "
                  f"MOTA none {m_none:.3f} -> opt {m_opt:.3f}  q90 {q:.3f}")
        if ratios:
            fitted = float(np.exp(np.mean(np.log(ratios))))
            result["kappa_fitted"] = fitted
            result["kappa_fit_sessions"] = len(ratios)
            result["train"] = per_train
            print(f"  => fitted kappa({args.label}) = {fitted:.3f} on {len(ratios)} sessions "
                  f"(spread {min(ratios):.2f}..{max(ratios):.2f})")

    # ---------- validation sessions ----------
    kappa_use = fitted if fitted else KAPPA_OURS
    ref_q90 = np.array([v["q90"] for v in result.get("train", {}).values()]) if result.get("train") else None
    rows = {}
    for v in VAL:
        if args.val_npy:
            per, hi = load_npy_cache(Path(args.val_npy), v)
        else:
            p = Path(args.val_dets) / f"official_{v}.txt"
            if not p.exists():
                print("  missing", p)
                continue
            per, hi = load_txt_dump(p)
        gt = parse_mot(Path(args.gt_val) / f"{v}.txt")
        gt = {f: b for f, b in gt.items() if 1 <= f <= hi}
        tr = one_pass(tracker_cls, per, hi, v)
        s_hat = kappa_use * clue[v]["clue"]
        m_none = mota_at(tr, gt, 1.0)
        m_l05 = mota_at(tr, gt, float(np.exp(0.5 * np.log(s_hat))))
        m_l10 = mota_at(tr, gt, float(np.clip(s_hat, 0.60, 1.60)))
        s_or, m_or = best_scale(tr, gt)
        q = q90(per)
        dyn = {}
        if ref_q90 is not None and len(ref_q90):
            ref, sd = float(ref_q90.mean()), float(ref_q90.std())
            for k in (2.0, 4.0, 6.0):
                lam = 0.5 + 0.5 * float(np.clip((ref - q) / (k * sd), 0, 1))
                dyn[k] = {"lambda": lam, "scale": float(np.exp(lam * np.log(s_hat))),
                          "mota": mota_at(tr, gt, float(np.exp(lam * np.log(s_hat))))}
        rows[v] = {"s_hat": s_hat, "q90": q, "mota_none": m_none, "mota_lambda05": m_l05,
                   "mota_lambda10": m_l10, "oracle_scale": s_or, "mota_oracle": m_or, "dynamic": dyn}
        print(f"  val {v}: s_hat {s_hat:.2f}  none {m_none:.4f}  lam0.5 {m_l05:.4f}  "
              f"lam1.0 {m_l10:.4f}  oracle({s_or:.2f}) {m_or:.4f}  "
              + " ".join(f"dyn k={k:g}:{d['mota']:.4f}" for k, d in dyn.items()))

    if rows:
        result["val"] = rows
        for key in ("mota_none", "mota_lambda05", "mota_lambda10", "mota_oracle"):
            result[f"mean_{key}"] = float(np.mean([r[key] for r in rows.values()]))
        for k in (2.0, 4.0, 6.0):
            vals = [r["dynamic"][k]["mota"] for r in rows.values() if k in r["dynamic"]]
            if vals:
                result[f"mean_dynamic_k{k:g}"] = float(np.mean(vals))
        print("\n  means:", {k: round(v, 4) for k, v in result.items() if k.startswith("mean_")})
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
