#!/usr/bin/env python3
"""Count how many tracks each tracker actually emits under strided detection.

This isolates *why* the published ByteTrack / BoT-SORT ports score 0.000 MOTA at
k=4 with the same detections: their unconfirmed tracks are removed on any call
where they are not matched, so under a detection-every-k-frames schedule a
newborn track dies in its first detection-free frame.  Counting emitted tracks
and rows is a tracker-independent diagnostic (no ground truth involved).

    python tracker_emission.py --strides 1,4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("tc", ROOT / "scripts" / "tracker_compare.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--strides", default="1,4")
    ap.add_argument("--out", default=str(ROOT / "data" / "tracker_emission.json"))
    args = ap.parse_args()

    makers = {
        "ours": lambda: tc.OursAdapter(),
        "bytetrack-default": lambda: tc.UltraAdapter("bytetrack", {}),
        "botsort-default": lambda: tc.UltraAdapter("botsort", {}),
        "ocsort-default": lambda: tc.OCSortAdapter({}),
    }
    out: dict[str, dict] = {}
    for k in [int(x) for x in args.strides.split(",") if x]:
        out[str(k)] = {}
        for v in [x for x in args.vids.split(",") if x]:
            pf, _ = tc.read_board(tc.find_dump(Path(tc.CODE / ".tmp" / "pruned_4sess"), v),
                                  tc.FW, tc.FH)
            hi = max(pf)
            out[str(k)][v] = {}
            for tag, mk in makers.items():
                tr = tc.run_session(mk(), pf, hi, 1, k, 1.0)
                out[str(k)][v][tag] = {
                    "tracks": len(tr),
                    "rows": int(sum(len(f) for f in tr.values())),
                }
                print(f"k={k} {v} {tag:18s} tracks={len(tr):4d} rows={out[str(k)][v][tag]['rows']:6d}")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
