#!/usr/bin/env python3
"""Statistics of the 20 VISEM-Tracking sessions used in the paper.

Reads the class-0 MOT ground truth (16 training sessions in
../data/gt_train, the 4 unseen sessions in ../../datasets/visem_tracking/
mot_gt_c0) and writes ../data/v2_dataset_stats.json for the dataset figure.

    python dataset_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # paper_visem
CODE = ROOT.parent                                     # D:\code
V2 = Path(__file__).resolve().parent

TRAIN = ["11", "12", "13", "14", "15", "19", "21", "22", "23", "24",
         "29", "30", "35", "36", "38", "47"]
VAL = ["82", "60", "54", "52"]
FW, FH = 640.0, 480.0


def gt_path(v: str) -> Path:
    if v in TRAIN:
        return ROOT / "data" / "gt_train" / f"{v}.txt"
    return CODE / "datasets" / "visem_tracking" / "mot_gt_c0" / f"{v}.txt"


def read_gt(v: str):
    per_frame: dict[int, int] = {}
    ids: set[int] = set()
    widths, heights = [], []
    for line in gt_path(v).read_text().splitlines():
        p = line.replace(",", " ").split()
        if len(p) < 6:
            continue
        f, i = int(float(p[0])), int(float(p[1]))
        w, h = float(p[4]) * FW, float(p[5]) * FH
        per_frame[f] = per_frame.get(f, 0) + 1
        ids.add(i)
        widths.append(w)
        heights.append(h)
    return per_frame, len(ids), np.array(widths), np.array(heights)


def main() -> int:
    out = {"train": [], "val": []}
    all_w = []
    for split, sessions in (("train", TRAIN), ("val", VAL)):
        for v in sessions:
            per_frame, n_ids, w, h = read_gt(v)
            frames = max(per_frame) if per_frame else 0
            counts = np.array(list(per_frame.values()))
            out[split].append({
                "session": v,
                "frames": int(frames),
                "targets": int(n_ids),
                "per_frame_mean": round(float(counts.mean()), 1),
                "per_frame_max": int(counts.max()),
                "box_w_median_px": round(float(np.median(w)), 2),
                "box_h_median_px": round(float(np.median(h)), 2),
            })
            all_w.append(w)
    w = np.concatenate(all_w)
    counts, edges = np.histogram(w, bins=40, range=(0, 80))
    out["summary"] = {
        "sessions": len(TRAIN) + len(VAL),
        "frames_total": int(sum(r["frames"] for r in out["train"] + out["val"])),
        "targets_total": int(sum(r["targets"] for r in out["train"] + out["val"])),
        "box_w_px_median": round(float(np.median(w)), 2),
        "box_w_px_p05": round(float(np.percentile(w, 5)), 2),
        "box_w_px_p95": round(float(np.percentile(w, 95)), 2),
        "gt_rows": int(w.size),
    }
    out["box_w_hist"] = {"edges": [round(float(e), 2) for e in edges],
                         "counts": [int(c) for c in counts]}
    (ROOT / "data" / "v2_dataset_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("summary:", out["summary"])
    for split in ("train", "val"):
        print(f"--- {split} ---")
        for r in out[split]:
            print(f"  {r['session']:>3}  frames {r['frames']:>5}  targets {r['targets']:>4} "
                  f" per-frame mean {r['per_frame_mean']:>5} max {r['per_frame_max']:>4}"
                  f"  w_med {r['box_w_median_px']:>5} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
