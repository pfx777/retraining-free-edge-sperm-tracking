#!/usr/bin/env python3
"""Does the *calibration* gain survive a third-party tracker?

The shipped pipeline applies a per-session output box scale (the label-free
calibration: kappa rule x precision offset x per-session lambda).  Everything
else - the detector, the frames, the association logic of each tracker - is held
fixed, and only that scale is switched between:

    s = 1                    (no calibration)
    FP32 rule transfer       (0.95 / 1.05 / 1.20 / 1.05   - what a naive port gives)
    deployed dynamic scales  (1.02 / 1.15 / 1.48 / 1.16   - the delivered recipe)

for four trackers, at k=1 and k=4, scored by the same class-0 MOTA.

    python calib_x_tracker.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tc", ROOT / "scripts" / "tracker_compare.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

VIDS = ("82", "60", "54", "52")
SCALE_MODES = {
    "no_calibration": {"82": 1.00, "60": 1.00, "54": 1.00, "52": 1.00},
    "fp32_rule_transfer": {"82": 0.95, "60": 1.05, "54": 1.20, "52": 1.05},
    "deployed_dynamic": tc.SCALES,
}
TRACKERS = {
    "ours": lambda: tc.OursAdapter(),
    "bytetrack": lambda: tc.UltraAdapter("bytetrack", {}),
    "botsort": lambda: tc.UltraAdapter("botsort", {}),
    "ocsort": lambda: tc.OCSortAdapter({}),
}


def main() -> int:
    out: dict = {}
    for k in (1, 4):
        out[str(k)] = {}
        print(f"\n=== k={k}  (board INT8 detections of the delivered model)")
        print(f"{'tracker':10s} " + " ".join(f"{m:>19s}" for m in SCALE_MODES) + "   Δ(部署−不校正)")
        for tag, maker in TRACKERS.items():
            row = {}
            for mode, scales in SCALE_MODES.items():
                per = {}
                for v in VIDS:
                    pf, _ = tc.read_board(tc.find_dump(tc.CODE / ".tmp" / "pruned_4sess", v),
                                          tc.FW, tc.FH)
                    gt = tc.parse_mot(tc.CODE / "datasets" / "visem_tracking" / "mot_gt_c0" / f"{v}.txt")
                    hi = max(pf)
                    gt = {f: b for f, b in gt.items() if 1 <= f <= hi}
                    tr = tc.run_session(maker(), pf, hi, 1, k, scales[v])
                    per[v] = tc.mota(gt, tc.interpolate(tr, 14))["MOTA"]
                row[mode] = {"per_session": per, "mean": float(np.mean(list(per.values())))}
            delta = row["deployed_dynamic"]["mean"] - row["no_calibration"]["mean"]
            row["delta_deployed_minus_none"] = delta
            out[str(k)][tag] = row
            cells = " ".join(f"{row[m]['mean']:19.4f}" for m in SCALE_MODES)
            print(f"{tag:10s} {cells}   {delta:+.4f}")

    (ROOT / "data" / "calib_x_tracker.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote data/calib_x_tracker.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
