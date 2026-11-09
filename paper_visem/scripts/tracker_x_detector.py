#!/usr/bin/env python3
"""Does the model-side gain survive a third-party tracker?

2x2 design at the deployed operating point (k=1, board INT8, per-session box
scale applied after tracking, one MOTA implementation):

    detector  x  tracker
    baseline stock YOLOv8n  x  ours / ByteTrack / BoT-SORT
    delivered E35 (gate+pruned) x  ours / ByteTrack / BoT-SORT

If the gain is a detection-quality effect, it must also appear with a tracker we
did not write.

    python tracker_x_detector.py
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
TRACKERS = {
    "ours": lambda: tc.OursAdapter(),
    "bytetrack": lambda: tc.UltraAdapter("bytetrack", {}),
    "botsort": lambda: tc.UltraAdapter("botsort", {}),
    "ocsort": lambda: tc.OCSortAdapter({}),
}
DUMP = {"baseline-det": "base_4sess", "e35-det": "pruned_4sess"}


def main() -> int:
    out: dict = {}
    for det_tag, folder in DUMP.items():
        out[det_tag] = {}
        for trk_tag, maker in TRACKERS.items():
            per = {}
            for v in VIDS:
                pf, _ = tc.read_board(tc.find_dump(tc.CODE / ".tmp" / folder, v), tc.FW, tc.FH)
                gt = tc.parse_mot(tc.CODE / "datasets" / "visem_tracking" / "mot_gt_c0" / f"{v}.txt")
                hi = max(pf)
                gt = {f: b for f, b in gt.items() if 1 <= f <= hi}
                tr = tc.run_session(maker(), pf, hi, 1, 1, tc.SCALES[v])
                per[v] = tc.mota(gt, tc.interpolate(tr, 14))["MOTA"]
            mean = float(np.mean(list(per.values())))
            out[det_tag][trk_tag] = {"per_session": per, "mean": mean}
            cells = " ".join(f"{per[v]:.4f}" for v in VIDS)
            print(f"{det_tag:12s} {trk_tag:10s} {cells}  mean {mean:.4f}")

    print("\ndelta (E35 - baseline) per tracker:")
    out["delta"] = {}
    for trk_tag in TRACKERS:
        d = out["e35-det"][trk_tag]["mean"] - out["baseline-det"][trk_tag]["mean"]
        out["delta"][trk_tag] = d
        print(f"  {trk_tag:10s} {d:+.4f}")
    (ROOT / "data" / "tracker_x_detector.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote data/tracker_x_detector.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
