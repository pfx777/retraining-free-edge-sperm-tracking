#!/usr/bin/env python3
"""Tracker-level comparison under one identical protocol.

Question: is the measured gain of our label-free calibration an artifact of the
SORT-style tracker we ship on the board, or does it survive with other
published trackers?

All four trackers consume **the same detections** (the board's INT8 dump of the
delivered pruned model), on **the same frames**, with **the same detection
schedule** (every k-th frame), the **same per-session output box scale** applied
after tracking (the board's semantics), and are scored by **the same MOTA**
(class-0 GT, IoU>=0.5 greedy matching, id switches charged, gaps<=14
interpolated) that every other number in this project uses.

Implementations used (all published code, unmodified):
  * ours          -- the exact Tracker class sliced out of rknn_headraw_pipeline.py
  * ByteTrack     -- ultralytics/trackers/byte_tracker.py (canonical port)
  * BoT-SORT      -- ultralytics/trackers/bot_sort.py
  * OC-SORT       -- third_party/ocsort (official repo, MIT)

Each tracker is run twice: once with its published default parameters and once
with thresholds matched to our operating point, so neither side is advantaged.

    python tracker_compare.py [--strides 1,4] [--vids 82,60,54,52]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # paper_visem/
CODE = ROOT.parent                                     # D:/code
# Keep ultralytics' config/caches inside the workspace (its default %APPDATA%
# path is not writable here) and provide the `lap` shim before any import.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "third_party" / "ultralytics_cfg"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(CODE / "ultralytics-8.3.163"))

from adaptive_sched_eval import load_board_tracker                    # noqa: E402
from score_board_dump_gctx import find_dump, read_board               # noqa: E402
from score_det_dump import interpolate, mota, parse_mot               # noqa: E402

FW, FH = 640.0, 480.0
SCALES = {"82": 1.02, "60": 1.15, "54": 1.48, "52": 1.16}   # deployed board scale

OURS = dict(track_high=0.60, new_track=0.60, track_low=0.05,
            match_iou=0.30, low_match_iou=0.70, max_age=60)


class Conf:
    """Minimal stand-in for an ultralytics Results object (conf / xywh / cls)."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls


def boxes_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """(x1,y1,x2,y2) -> (cx,cy,w,h)"""
    out = boxes.copy().astype(np.float64)
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    out[:, 0] = boxes[:, 0] + out[:, 2] / 2
    out[:, 1] = boxes[:, 1] + out[:, 3] / 2
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy().astype(np.float64)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


# --------------------------------------------------------------------------- adapters


class OursAdapter:
    name = "ours (SORT/Kalman, shipped on board)"

    def __init__(self):
        Tracker = load_board_tracker(CODE / "rknn_headraw_pipeline.py")
        self.tr = Tracker(OURS["track_high"], OURS["new_track"], OURS["track_low"],
                          OURS["match_iou"], OURS["low_match_iou"], OURS["max_age"])

    def update(self, dets: np.ndarray):
        """dets: (N,5) cx,cy,w,h,score (pixels). Returns (M,5) cx,cy,w,h,id."""
        payload = [{"xywh": d[:4].astype(np.float32), "conf": float(d[4])} for d in dets]
        self.tr.update(payload)
        rows = []
        for k in range(len(self.tr.xywh)):
            if self.tr.age[k] != 0:      # same rule as the board dump
                continue
            cx, cy, w, h = self.tr.xywh[k]
            rows.append((cx, cy, w, h, int(self.tr.ids[k])))
        return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


class UltraAdapter:
    """ByteTrack or BoT-SORT from the ultralytics package."""

    def __init__(self, kind: str, params: dict):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT

        base = dict(track_high_thresh=0.50, track_low_thresh=0.10, new_track_thresh=0.60,
                    track_buffer=30, match_thresh=0.80, fuse_score=True, frame_rate=25)
        if kind == "botsort":
            base.update(proximity_thresh=0.50, appearance_thresh=0.25,
                        with_reid=False, model="auto", gmc_method="sparseOptFlow")
        base.update(params)
        self.args = SimpleNamespace(**base)
        cls = BYTETracker if kind == "bytetrack" else BOTSORT
        self.tr = cls(self.args, frame_rate=base["frame_rate"])
        self.name = "ByteTrack (ultralytics)" if kind == "bytetrack" else "BoT-SORT (ultralytics)"

    def update(self, dets: np.ndarray):
        if len(dets):
            xywh = dets[:, :4].astype(np.float32)
            conf = dets[:, 4].astype(np.float32)
            cls = np.zeros(len(dets), dtype=np.float32)
        else:
            xywh = np.zeros((0, 4), np.float32)
            conf = np.zeros((0,), np.float32)
            cls = np.zeros((0,), np.float32)
        out = self.tr.update(Conf(xywh, conf, cls), img=None)
        if out is None or len(out) == 0:
            return np.zeros((0, 5))
        out = np.asarray(out, dtype=np.float64)
        boxes = out[:, :4]                    # x1,y1,x2,y2
        ids = out[:, 4]
        xywh = boxes_to_xywh(boxes)
        return np.column_stack([xywh, ids])


class GapAwareUltra(UltraAdapter):
    """ByteTrack / BoT-SORT with one documented policy change for strided detection.

    The published implementations delete every unconfirmed track on any call in
    which it is not matched.  With a detector that runs every k-th frame, the
    next call has no detections at all, so every newborn track is deleted before
    it can ever be confirmed (measured: 7-23 emitted tracks at k=4 instead of
    187-474).  This subclass keeps unconfirmed tracks alive **only on frames
    where the detector produced no detections**, which is the minimal change
    that makes the tracker usable in the strided regime.  Behaviour on frames
    with detections is untouched, so the k=1 results are identical.
    """

    def update(self, dets: np.ndarray):
        if len(dets) == 0:
            from ultralytics.trackers.byte_tracker import STrack

            original = STrack.mark_removed
            STrack.mark_removed = lambda self: None  # noqa: ARG005
            try:
                return super().update(dets)
            finally:
                STrack.mark_removed = original
        return super().update(dets)


class OCSortAdapter:
    name = "OC-SORT (official repo)"

    def __init__(self, params: dict):
        from ocsort.ocsort import OCSort

        cfg = dict(det_thresh=0.30, max_age=30, min_hits=3, iou_threshold=0.30,
                   delta_t=3, asso_func="iou", inertia=0.2, use_byte=False)
        cfg.update(params)
        self.tr = OCSort(det_thresh=cfg["det_thresh"], max_age=cfg["max_age"],
                         min_hits=cfg["min_hits"], iou_threshold=cfg["iou_threshold"],
                         delta_t=cfg["delta_t"], asso_func=cfg["asso_func"],
                         inertia=cfg["inertia"], use_byte=cfg["use_byte"])

    def update(self, dets: np.ndarray):
        if len(dets):
            xyxy = xywh_to_xyxy(dets[:, :4])
            arr = np.column_stack([xyxy, dets[:, 4]]).astype(np.float64)
        else:
            arr = np.zeros((0, 5))
        out = self.tr.update(arr, (FH, FW), (FH, FW))   # scale becomes 1.0
        if out is None or len(out) == 0:
            return np.zeros((0, 5))
        out = np.asarray(out, dtype=np.float64)
        boxes = out[:, :4]
        ids = out[:, 4]
        return np.column_stack([boxes_to_xywh(boxes), ids])


# --------------------------------------------------------------------------- experiment


def run_session(adapter, per_frame, hi, lo, k, scale):
    """Feed the schedule to one tracker; return {id: {frame: (cx,cy,w,h)}}."""
    tracks: dict[int, dict[int, tuple]] = {}
    for fr in range(lo, hi + 1):
        rows = per_frame.get(fr, []) if (fr - lo) % k == 0 else []
        dets = np.asarray([[cx, cy, w, h, s] for _lv, s, (cx, cy, w, h) in rows], dtype=np.float64)
        if dets.size == 0:
            dets = np.zeros((0, 5))
        out = adapter.update(dets)
        for cx, cy, w, h, tid in out:
            tracks.setdefault(int(tid), {})[fr] = (cx / FW, cy / FH, w * scale / FW, h * scale / FH)
    return tracks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", default=str(CODE / ".tmp" / "pruned_4sess"))
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--strides", default="1,4")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--gap", type=int, default=14)
    ap.add_argument("--gt-dir", default=str(CODE / "datasets" / "visem_tracking" / "mot_gt_c0"))
    ap.add_argument("--out", default=str(ROOT / "data" / "tracker_compare.json"))
    args = ap.parse_args()

    vids = [v for v in args.vids.split(",") if v]
    strides = [int(x) for x in args.strides.split(",") if x]

    variants = [
        ("ours-default", lambda: OursAdapter()),
        ("bytetrack-default", lambda: UltraAdapter("bytetrack", {})),
        ("bytetrack-matched", lambda: UltraAdapter("bytetrack", dict(
            track_high_thresh=0.60, new_track_thresh=0.60, track_low_thresh=0.05,
            track_buffer=60, match_thresh=0.70))),
        ("botsort-default", lambda: UltraAdapter("botsort", {})),
        ("botsort-matched", lambda: UltraAdapter("botsort", dict(
            track_high_thresh=0.60, new_track_thresh=0.60, track_low_thresh=0.05,
            track_buffer=60, match_thresh=0.70))),
        ("ocsort-default", lambda: OCSortAdapter({})),
        ("ocsort-matched", lambda: OCSortAdapter(dict(
            det_thresh=0.60, max_age=60, min_hits=1, iou_threshold=0.30))),
        ("bytetrack-gapaware", lambda: GapAwareUltra("bytetrack", {})),
        ("botsort-gapaware", lambda: GapAwareUltra("botsort", {})),
    ]

    data = {}
    for v in vids:
        pf, _rows = read_board(find_dump(Path(args.dump_root), v), FW, FH)
        gt = parse_mot(Path(args.gt_dir) / f"{v}.txt")
        hi = max(pf)
        gt = {f: b for f, b in gt.items() if args.lo <= f <= hi}
        data[v] = (pf, gt, hi)
        print(f"{v}: frames {args.lo}..{hi}, detections {sum(len(x) for x in pf.values())}")

    results = {}
    for k in strides:
        print(f"\n=== stride k={k} (detector runs every {k}-th frame)")
        print(f"{'tracker':28s} " + " ".join(f"{v:>7s}" for v in vids) + "     mean")
        results[k] = {}
        for tag, factory in variants:
            per_vid = {}
            name = None
            for v in vids:
                pf, gt, hi = data[v]
                # A tracker instance is stateful: it must be fresh for every
                # session, otherwise tracks from the previous video leak in.
                adapter = factory()
                name = adapter.name
                tr = run_session(adapter, pf, hi, args.lo, k, SCALES[v])
                per_vid[v] = mota(gt, interpolate(tr, args.gap))["MOTA"]
            mean = float(np.mean(list(per_vid.values())))
            results[k][tag] = {"per_session": per_vid, "mean": mean, "tracker": name}
            print(f"{(name or tag)[:28]:28s} " + " ".join(f"{per_vid[v]:7.4f}" for v in vids)
                  + f"  {mean:7.4f}")

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
