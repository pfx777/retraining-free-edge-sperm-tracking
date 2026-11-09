#!/usr/bin/env python3
"""Per-video BOX_SCALE calibration sweep.

Why this exists
---------------
Global box calibration is provably impossible on this dataset.  The detection
box "size ratio" (median det/GT side over matched pairs) sits on opposite sides
of 1.0 for different videos, and for the best model even the interquartile
ranges are disjoint:

    ablation_lkctx30   video 54   ratio 0.884  [0.812, 0.959]
                       video 60   ratio 0.880  [0.775, 0.997]
                       video 82   ratio 1.172  [1.041, 1.316]   <- p25 > 54's p75

So no single BOX_SCALE can bring both near 1.0, and the IoU-0.5 cliff at 0.707
turns the residual into lost MOTA.  This script measures what a *per-video*
scale is worth, and where each video's optimum sits.

First measured on ablation_lkctx30 @640 (0.539 global mean):

    BOX_SCALE    video 54   video 60   video 82
       0.85        0.105      0.151      0.722   <- 82 peaks at the grid edge
       0.95        0.466      0.452      0.634
       1.00        0.548      0.545      0.523
       1.05        0.602      0.608      0.418
       1.15        0.628      0.661      0.137   <- 54/60 still rising

    per-video best mean = (0.628 + 0.661 + 0.722) / 3 = 0.670

Both gains matter: scaling up on 54/60 cuts FP (10908->9232, 2658->1534) as well
as FN, because better-fitting boxes let NMS suppress more duplicates.

Deployment note: this is a *calibration* protocol, not an inference-time trick.
A fixed microscope rig is calibrated once per session, so a per-video scale is
legitimate -- but the paper must say so, and validate it on the sealed holdout.

    python sweep_per_video_scale.py --run-name ablation_lkctx30 \
        --scales 0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25

    # same sweep for the small deployment candidate at two input sizes
    python sweep_per_video_scale.py --run-name ablation_dyhead30 \
        --sizes 640,448 --scales 0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/code")
REID = "/root/code/runs/visem_official/sperm_reid_v2.pt"
TRACK_ROOT = "/root/code/runs/visem_tracking"

RUNS_PARENTS = (
    "/root/autodl-tmp/runs/detect/runs/visem_official",
    "/root/code/runs/detect/runs/visem_official",
    "/root/autodl-tmp/runs/detect/runs/visem_tracking",
    "/root/code/runs/detect/runs/visem_tracking",
)

TRACK_ENV_COMMON = {
    "CONF": "0.02",
    "NMS_IOU": "0.5",
    "MAX_DET": "2000",
    "TRACK_HIGH": "0.50",
    "NEW_TRACK": "0.50",
    "TRACK_LOW": "0.05",
    "LOW_MATCH_IOU": "0.50",
    "MATCH_IOU": "0.3",
    "MIN_HITS": "1",
    "MAX_AGE": "60",
    "LAMBDA_APP": "0.3",
}

MOTA_RE = re.compile(
    r"gap=14:\s*Mean MOTA=(-?[\d.]+)\s+Aggregate MOTA=(-?[\d.]+)\s+FP=(\d+)\s+FN=(\d+)\s+IDSW=(\d+)"
)


def stream(cmd: list[str], extra_env: dict[str, str]) -> str:
    env = {**os.environ, **extra_env}
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        chunks.append(line)
    return "".join(chunks)


def resolve_best(run_name: str) -> Path | None:
    for parent in RUNS_PARENTS:
        cand = Path(parent) / run_name / "weights" / "best.pt"
        if cand.is_file():
            return cand
    return None


def run_one(best: Path, vid: str, imgsz: int, scale: str, tag: str) -> dict[str, float]:
    env = {
        **TRACK_ENV_COMMON,
        "MODEL": str(best),
        "REID_MODEL": REID,
        "VAL_VIDS": vid,
        "BOX_SCALE": scale,
        "IMGSZ": str(imgsz),
        "OUT_DIR": f"{TRACK_ROOT}/pvscale_{tag}",
    }
    m = MOTA_RE.search(stream([sys.executable, "run_tracking_interp_sweep.py", "--gaps", "14"], env))
    if not m:
        return {}
    return {"MOTA": float(m.group(1)), "FP": int(m.group(3)), "FN": int(m.group(4)), "IDSW": int(m.group(5))}


def sweep_size(best: Path, vids: list[str], scales: list[str], imgsz: int) -> dict[str, dict[str, dict[str, float]]]:
    table: dict[str, dict[str, dict[str, float]]] = {v: {} for v in vids}
    for vid in vids:
        for sc in scales:
            tag = f"{best.parent.parent.name}_{imgsz}_{vid}_{sc.replace('.', '')}"
            res = run_one(best, vid, imgsz, sc, tag)
            table[vid][sc] = res
            if res:
                print(f"  imgsz={imgsz} video={vid:>3} scale={sc:<5} MOTA={res['MOTA']:.3f} "
                      f"FP={res['FP']} FN={res['FN']} IDSW={res['IDSW']}", flush=True)
            else:
                print(f"  imgsz={imgsz} video={vid:>3} scale={sc:<5} PARSE FAILED", flush=True)
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None, help="run directory under a runs parent")
    ap.add_argument("--model", default=None, help="explicit best.pt path (overrides --run-name)")
    ap.add_argument("--sizes", default="640", help="input sizes, comma separated")
    ap.add_argument("--scales", default="0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25")
    ap.add_argument("--vids", default="82,60,54,52",
                    help="full validation split by default (52 is no longer held out)")
    ap.add_argument("--baseline-scale", default="1.00")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    best = Path(args.model) if args.model else (resolve_best(args.run_name) if args.run_name else None)
    if best is None or not best.is_file():
        print("no best.pt found; pass --run-name or --model")
        return 1
    name = args.run_name or best.parent.parent.name
    print(f"run      : {name}")
    print(f"weights  : {best}")
    print(f"started  : {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    vids = [v.strip() for v in args.vids.split(",") if v.strip()]

    rows: list[dict[str, object]] = []
    for imgsz in sizes:
        print("=" * 78)
        print(f"### imgsz={imgsz}")
        table = sweep_size(best, vids, scales, imgsz)

        print(f"\nMOTA over BOX_SCALE   (imgsz={imgsz})")
        print("  video  " + "".join(f"{s:>8}" for s in scales) + f"{'best':>8}{'@scale':>8}")
        best_per_video: dict[str, tuple[str, float]] = {}
        for vid in vids:
            cells, best_sc, best_val = [], None, float("-inf")
            for sc in scales:
                val = table[vid].get(sc, {}).get("MOTA")
                cells.append(f"{val:>8.3f}" if val is not None else f"{'-':>8}")
                if val is not None and val > best_val:
                    best_sc, best_val = sc, val
            best_per_video[vid] = (best_sc, best_val)
            print(f"  {vid:>5}  " + "".join(cells) + f"{best_val:>8.3f}{best_sc:>8}")

        base_vals = [table[v].get(args.baseline_scale, {}).get("MOTA") for v in vids]
        base_valid = [x for x in base_vals if x is not None]
        # Only meaningful when the baseline scale is actually in the grid; a
        # partial grid (e.g. --vids 82 --scales 0.65,...,0.85) has no global
        # reference and must not silently report 0.000.
        base_mean = sum(base_valid) / len(base_valid) if base_valid else None
        best_mean = sum(x for _, x in best_per_video.values()) / len(vids)
        if base_mean is None:
            print(f"\n  (BOX_SCALE={args.baseline_scale} not in --scales: no global baseline for this grid)")
        else:
            print(f"\n  global BOX_SCALE={args.baseline_scale}  dev mean = {base_mean:.3f}")
        delta = f"   (+{best_mean - base_mean:.3f})" if base_mean is not None else ""
        print(f"  per-video best          dev mean = {best_mean:.3f}{delta}")
        for vid in vids:
            print(f"    video {vid}: scale={best_per_video[vid][0]}  MOTA={best_per_video[vid][1]:.3f}")

        for vid in vids:
            rows.append({
                "imgsz": imgsz, "video": vid,
                "best_scale": best_per_video[vid][0], "best_mota": round(best_per_video[vid][1], 4),
                "global_scale": args.baseline_scale,
                "global_mota": round(table[vid].get(args.baseline_scale, {}).get("MOTA", float("nan")), 4),
                **{f"mota@{s}": round(table[vid].get(s, {}).get("MOTA", float("nan")), 4) for s in scales},
            })
        rows.append({"imgsz": imgsz, "video": "MEAN", "best_scale": "-",
                     "best_mota": round(best_mean, 4), "global_scale": args.baseline_scale,
                     "global_mota": (round(base_mean, 4) if base_mean is not None else "")})

    out = Path(args.out) if args.out else ROOT / "runs/visem_tracking" / f"pvscale_sweep_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
