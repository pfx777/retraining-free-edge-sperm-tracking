#!/usr/bin/env python3
"""Isolate the tracker half of the board's deployment gap.

The board's INT8 pipeline scores 0.6529 MOTA on video 82 frames 1-300 where the
host's FP32 pipeline scores 0.7543 on the same frames.  That gap is not pure
quantisation: the board script carries its own simplified tracker (a
constant-velocity model over flat arrays) instead of the host's DeepOCSORT, and
the frames reach it as JPEGs rather than through the video decoder.

This script removes the tracker from the suspect list.  It runs the HOST detector
over the same 300 frames and feeds the detections through the VERY SAME tracker
class the board uses -- imported out of `rknn_headraw_pipeline.py` rather than
reimplemented, so the comparison cannot drift -- then scores the result with the
same MOTA code.

  host detector + board tracker  vs  host detector + host tracker (0.7543)
      -> the tracker's share
  board detector + board tracker (0.6529) minus that
      -> what is left for INT8 quantisation (and the JPEG re-encode)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/root/code")
sys.path.insert(0, str(ROOT))


def load_board_tracker(path: Path):
    """Exec the tracker section of the board script in a fresh namespace.

    Slicing rather than copying is deliberate: a copy would silently diverge the
    next time the board script changes, and then the comparison would measure the
    copy instead of the thing that actually ran.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # locate by content, not by line number: the file changes and a stale slice
    # would silently compare against a truncated copy of the class
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def iou_xywh"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("def main()"))
    body = "".join(lines[start:end])
    ns: dict = {"np": np}
    exec(compile(body, str(path), "exec"), ns)
    return ns["Tracker"], ns["iou_xywh"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vid", default="82")
    ap.add_argument("--frames-dir", default="/root/code/deploy_check/82_frames")
    ap.add_argument("--board-script", default="/root/code/rknn_headraw_pipeline.py")
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--box-scale", type=float, default=0.80)
    ap.add_argument("--dump-dir", default="/root/code/deploy_check/hostdet_boardtrk_82")
    ap.add_argument("--tracker", choices=("board", "host"), default="board",
                    help="board = the simplified tracker inside rknn_headraw_pipeline.py "
                         "(what produced 0.7081); host = the real DeepOCSORT the deployed "
                         "pipeline runs.  Running both on the SAME host detections and the "
                         "SAME frames isolates the tracker from the frame source and from "
                         "quantisation.")
    args = ap.parse_args()

    import cv2
    import mmsc_modules  # noqa: F401
    from ultralytics import YOLO
    import run_tracking_deepocsort as rtd  # noqa: E402
    import run_tracking_interp_sweep as interp  # noqa: E402

    if args.tracker == "board":
        Tracker, _iou = load_board_tracker(Path(args.board_script))
        tracker = Tracker(0.60, 0.60, 0.05, 0.30, 0.70, 60)
    else:
        # the deployed operating point, and no appearance term so that a missing
        # ReID feature is genuinely unused rather than silently compared
        rtd.TRACK_HIGH = 0.60
        rtd.NEW_TRACK = 0.60
        rtd.TRACK_LOW = 0.05
        rtd.LOW_MATCH_IOU = 0.70
        rtd.MATCH_IOU = 0.30
        rtd.MIN_HITS = 1
        rtd.MAX_AGE = 60
        rtd.LAMBDA_APP = 0.0
        tracker = rtd.Tracker()

    model = YOLO("/root/autodl-tmp/runs/detect/runs/visem_official/"
                 "ablation_nop2448/weights/best.pt")
    paths = sorted(Path(args.frames_dir).glob("*.jpg"))
    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)

    for fi, p in enumerate(paths, start=1):
        frame = cv2.imread(str(p))
        fh, fw = frame.shape[:2]
        r = model.predict(frame, conf=args.conf, iou=args.iou, max_det=2000,
                          imgsz=args.imgsz, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xywh = r.boxes.xywh.cpu().numpy().astype(np.float32)
            conf = r.boxes.conf.cpu().numpy().astype(np.float32)
            for k in range(len(conf)):
                dets.append({"xywh": xywh[k], "conf": float(conf[k]), "feat": None})
        if args.tracker == "board":
            tracker.update(dets)
            rows = [(tracker.xywh[k], int(tracker.ids[k]), float(tracker.conf[k]))
                    for k in range(len(tracker.xywh)) if tracker.age[k] == 0]
        else:
            live = tracker.update(dets)
            rows = [(t.xywh, int(t.id), 1.0) for t in live]
        with (out / f"{fi}.txt").open("w") as fh_out:
            for box, tid, tconf in rows:
                cx, cy, bw, bh = box
                fh_out.write(f"0 {cx / fw:.6f} {cy / fh:.6f} "
                             f"{bw * args.box_scale / fw:.6f} "
                             f"{bh * args.box_scale / fh:.6f} "
                             f"{tid} {tconf:.4f}\n")
        if fi % 100 == 0:
            print(f"  {fi} frames", flush=True)

    tracks = defaultdict(dict)
    for p in sorted(out.glob("*.txt")):
        fr = int(p.stem)
        for line in p.read_text().splitlines():
            q = line.split()
            if len(q) >= 6:
                tracks[int(float(q[5]))][fr] = (float(q[1]), float(q[2]),
                                                float(q[3]), float(q[4]))
    gt = rtd.parse_mot(str(rtd.MOT_GT / f"{args.vid}.txt"))
    gt = {f: b for f, b in gt.items() if 1 <= f <= len(paths)}
    frames = interp.interpolate_tracks(tracks, 14)
    pred = defaultdict(list)
    for fr, items in frames.items():
        for tid, box in items:
            pred[fr].append({"id": tid, "bbox": list(box)})
    m = rtd.mot_metrics(gt, pred)
    print(f"\nhost detector + {args.tracker.upper()} tracker, "
          f"video {args.vid} frames 1..{len(paths)}")
    print(f"  MOTA {m['MOTA']:.4f}  FP {m['FP']}  FN {m['FN']}  IDSW {m['IDSW']}  TP {m['TP']}")
    print(f"  recall {m['TP']/max(m['TP']+m['FN'],1):.4f}  "
          f"precision {m['TP']/max(m['TP']+m['FP'],1):.4f}")
    print("  (host tracker reference on the same frames: MOTA 0.7543, "
          "recall 0.8302, precision 0.9170)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
