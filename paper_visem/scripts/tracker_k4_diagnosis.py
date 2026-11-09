#!/usr/bin/env python3
"""Pin down *why* ByteTrack/BoT-SORT collapse at k=4, frame by frame.

Reports, per session and detection stride:
  n_dets        detections handed to the tracker on that call
  tracked       tracks in Tracked state (activated or not)
  activated     subset with is_activated == True
  unconfirmed   newborn tracks that have not been confirmed yet
  lost          tracks in Lost state
  out           rows the tracker returned

and the MOTA confusion counts, so the drop can be attributed to FN/FP/IDSW.

Two hypotheses are tested side by side:
  H1 (policy)   the published code deletes unconfirmed tracks that are not
                matched on the current call.  Test: make the unconfirmed step
                never report "unmatched" (a 5-line, clearly scoped monkey-patch
                on matching.linear_assignment when thresh == 0.7), so newborn
                tracks survive detection-free stretches.
  H2 (geometry) even with H1 patched, a newborn track's Kalman state has zero
                velocity, so its prediction after 4 frames is the same box and
                the IoU gate fails in fast motion.  Test: after H1, check
                whether tracks now survive and are matched later.

    python tracker_k4_diagnosis.py --vids 82,60 --strides 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tc", ROOT / "scripts" / "tracker_compare.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


class InstrumentedUltra(tc.UltraAdapter):
    """ByteTrack / BoT-SORT with per-call state logging."""

    def __init__(self, kind: str, params: dict, keep_unconfirmed: bool = False):
        super().__init__(kind, params)
        self.log: list[dict] = []
        self.keep_unconfirmed = keep_unconfirmed
        self._patched = None

    def _patch_matching(self):
        import ultralytics.trackers.utils.matching as matching

        original = matching.linear_assignment

        def patched(cost_matrix, thresh):
            matches, u_track, u_det = original(cost_matrix, thresh)
            if abs(float(thresh) - 0.7) < 1e-9:
                # The unconfirmed step: do not declare a newborn track unmatched.
                return matches, [], u_det
            return matches, u_track, u_det

        matching.linear_assignment = patched
        self._patched = (matching, original)

    def _unpatch(self):
        if self._patched is not None:
            module, original = self._patched
            module.linear_assignment = original
            self._patched = None

    def update(self, dets: np.ndarray):
        if self.keep_unconfirmed:
            self._patch_matching()
        try:
            out = super().update(dets)
        finally:
            self._unpatch()
        tr = self.tr
        tracked = list(getattr(tr, "tracked_stracks", []))
        lost = list(getattr(tr, "lost_stracks", []))
        self.log.append({
            "frame_call": tr.frame_id,
            "n_dets": int(len(dets)),
            "tracked": len(tracked),
            "activated": int(sum(1 for t in tracked if getattr(t, "is_activated", False))),
            "unconfirmed": int(sum(1 for t in tracked if not getattr(t, "is_activated", False))),
            "lost": len(lost),
            "out": int(len(out)),
        })
        return out


def summarise(log: list[dict]) -> dict:
    if not log:
        return {}
    det_calls = [r for r in log if r["n_dets"] > 0]
    return {
        "calls": len(log),
        "det_calls": len(det_calls),
        "max_tracked": max(r["tracked"] for r in log),
        "max_activated": max(r["activated"] for r in log),
        "max_unconfirmed": max(r["unconfirmed"] for r in log),
        "final_tracked": log[-1]["tracked"],
        "final_activated": log[-1]["activated"],
        "out_rows": int(sum(r["out"] for r in log)),
        "first_call": log[0],
        "second_det_call": det_calls[1] if len(det_calls) > 1 else None,
        "sample_mid": det_calls[len(det_calls) // 2] if det_calls else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vids", default="82,60")
    ap.add_argument("--strides", default="4")
    ap.add_argument("--dump-root", default=str(tc.CODE / ".tmp" / "pruned_4sess"))
    ap.add_argument("--gt-dir", default=str(tc.CODE / "datasets" / "visem_tracking" / "mot_gt_c0"))
    ap.add_argument("--out", default=str(ROOT / "data" / "tracker_k4_diagnosis.json"))
    args = ap.parse_args()

    out: dict = {}
    for k in [int(x) for x in args.strides.split(",") if x]:
        out[str(k)] = {}
        for v in [x for x in args.vids.split(",") if x]:
            pf, _ = tc.read_board(tc.find_dump(Path(args.dump_root), v), tc.FW, tc.FH)
            gt = tc.parse_mot(Path(args.gt_dir) / f"{v}.txt")
            hi = max(pf)
            gt = {f: b for f, b in gt.items() if 1 <= f <= hi}
            row: dict = {}
            for tag, maker in (
                ("ours", lambda: tc.OursAdapter()),
                ("bytetrack-default", lambda: InstrumentedUltra("bytetrack", {})),
                ("bytetrack-keep-unconfirmed", lambda: InstrumentedUltra("bytetrack", {}, True)),
                ("botsort-default", lambda: InstrumentedUltra("botsort", {})),
                ("botsort-keep-unconfirmed", lambda: InstrumentedUltra("botsort", {}, True)),
            ):
                ad = maker()
                tr = tc.run_session(ad, pf, hi, 1, k, tc.SCALES[v])
                stats = tc.mota(gt, tc.interpolate(tr, 14))
                entry = {"mota": stats["MOTA"], "TP": stats["TP"], "FP": stats["FP"],
                         "FN": stats["FN"], "IDSW": stats["IDSW"], "GT": stats["GT"],
                         "tracks": len(tr)}
                if hasattr(ad, "log"):
                    entry["state"] = summarise(ad.log)
                row[tag] = entry
                print(f"k={k} {v} {tag:28s} MOTA {stats['MOTA']:+.4f}  TP {stats['TP']:6d} "
                      f"FN {stats['FN']:6d} FP {stats['FP']:6d} IDSW {stats['IDSW']:5d} "
                      f"tracks {len(tr):4d}")
            out[str(k)][v] = row

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", args.out)
    # Print the state trace of the interesting cases for the record.
    for k, vids in out.items():
        for v, row in vids.items():
            for tag in ("bytetrack-default", "bytetrack-keep-unconfirmed"):
                st = row.get(tag, {}).get("state")
                if st:
                    print(f"\n{k}/{v} {tag}: calls={st['calls']} det_calls={st['det_calls']} "
                          f"max_tracked={st['max_tracked']} max_activated={st['max_activated']} "
                          f"max_unconfirmed={st['max_unconfirmed']} final_tracked={st['final_tracked']}")
                    print("   first call:", json.dumps(st["first_call"], ensure_ascii=False))
                    print("   2nd det call:", json.dumps(st["second_det_call"], ensure_ascii=False))
                    print("   mid det call:", json.dumps(st["sample_mid"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
