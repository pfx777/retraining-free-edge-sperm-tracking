#!/usr/bin/env python3
"""Board-side A/B pipeline: same model, same tracker maths, four cost profiles.

The reference `rknn_headraw_pipeline.py` reports preprocess / NPU / decode+NMS /
tracker, but two real costs never appear in those numbers:

  * the per-detection conversion loop (one `np.array` per box, ~35 boxes/frame),
  * the tracker's re-stacking of those per-detection arrays.

This script runs the same work with the switches exposed, so the board can say
which one actually costs the milliseconds:

    --post ref|fast     reference decode+NMS   vs  the low-dispatch fastpost
    --dets dict|array   per-detection dicts    vs  one (D,4) array + (D,) scores

Every stage is timed, including the conversion loop that the reference leaves
untimed, and the end line prints the *complete* per-frame cost so the FPS number
can be compared with the wall clock.

    python3 rknn_headraw_pipeline_fast.py --rknn stock_yolov8n_448_int8.rknn \
        --imgsz 448 --video $V/82/82.mp4 --limit 1500 --stride 1 \
        --conf 0.02 --iou 0.5 --track-high 0.60 --new-track 0.60 --track-low 0.05 \
        --match-iou 0.30 --low-match-iou 0.70 --max-age 60 --box-scale 0.90 \
        --post fast --dets array
"""

import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np

import fastpost
import rknn_headraw_detect as refdet
from fasttrack import TrackerArrays
from rknn_headraw_pipeline import Tracker as RefTracker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", required=True)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--video", default=None)
    ap.add_argument("--img_dir", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--track-high", type=float, default=0.60)
    ap.add_argument("--new-track", type=float, default=0.60)
    ap.add_argument("--track-low", type=float, default=0.05)
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--low-match-iou", type=float, default=0.70)
    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--box-scale", type=float, default=1.0)
    ap.add_argument("--post", choices=("ref", "fast"), default="fast")
    ap.add_argument("--dets", choices=("dict", "array"), default="array")
    ap.add_argument("--pre", choices=("ref", "fast"), default="ref",
                    help="ref: reuse one canvas, memset it 114 then copy the ROI in "
                         "(reference); fast: copyMakeBorder straight into a fresh canvas")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--counts", action="store_true",
                    help="print candidates/kept per detected frame (mean)")
    args = ap.parse_args()

    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    print("load_rknn:", rknn.load_rknn(args.rknn))
    print("init_runtime:", rknn.init_runtime(core_mask=0))

    if args.video:
        cap = cv2.VideoCapture(args.video)
        frames = []
        while len(frames) < args.limit:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        name = os.path.basename(args.video)
    elif args.img_dir:
        paths = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))[: args.limit]
        frames = [cv2.imread(p) for p in paths]
        frames = [f for f in frames if f is not None]
        name = args.img_dir
    else:
        raise SystemExit("pass --video or --img_dir")
    if not frames:
        raise SystemExit("no frames")

    if args.dets == "array":
        tracker = TrackerArrays(args.track_high, args.new_track, args.track_low,
                                args.match_iou, args.low_match_iou, args.max_age)
    else:
        tracker = RefTracker(args.track_high, args.new_track, args.track_low,
                             args.match_iou, args.low_match_iou, args.max_age)

    t_pre = t_inf = t_sco = t_dfl = t_nms = t_bld = t_trk = 0.0
    n_det = n_out = 0
    n_cand = n_kept = n_rows = 0
    n_tracks = 0
    canvas = np.full((args.imgsz, args.imgsz, 3), 114, dtype=np.uint8)
    print(f"\n{name}: {len(frames)} frames, detector stride {args.stride}, "
          f"post {args.post}, dets {args.dets}")

    wall0 = time.perf_counter()
    for fi, frame in enumerate(frames):
        acc = fi >= args.warmup
        s = time.perf_counter()
        fh, fw = frame.shape[:2]
        r = args.imgsz / max(fh, fw)
        nh, nw = max(1, int(round(fh * r))), max(1, int(round(fw * r)))
        dx, dy = (args.imgsz - nw) // 2, (args.imgsz - nh) // 2
        if args.pre == "fast":
            # one border copy instead of a full-canvas memset plus an ROI copy
            canvas = cv2.cvtColor(
                cv2.copyMakeBorder(cv2.resize(frame, (nw, nh),
                                              interpolation=cv2.INTER_LINEAR),
                                   dy, args.imgsz - nh - dy, dx, args.imgsz - nw - dx,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114)),
                cv2.COLOR_BGR2RGB)
        else:
            small = cv2.cvtColor(cv2.resize(frame, (nw, nh),
                                            interpolation=cv2.INTER_LINEAR),
                                 cv2.COLOR_BGR2RGB)
            canvas[:] = 114
            canvas[dy:dy + nh, dx:dx + nw] = small
        d_pre = time.perf_counter() - s

        boxes = np.zeros((0, 4), np.float32)
        kept = np.zeros((0,), np.float32)
        d_inf = d_sco = d_dfl = d_nms = 0.0
        if fi % args.stride == 0:
            s = time.perf_counter()
            outs = rknn.inference(inputs=[canvas[None, ...]], data_format="nhwc")
            d_inf = time.perf_counter() - s
            if args.post == "ref":
                s = time.perf_counter()
                scores, _shapes = refdet.class_scores_fast(outs)
                keep = np.flatnonzero(scores > args.conf)
                d_sco = time.perf_counter() - s
                if keep.size:
                    s = time.perf_counter()
                    cxcywh = refdet.boxes_for(outs, args.imgsz, keep)
                    cx, cy, w, h = cxcywh
                    cand = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
                    cand_sc = scores[keep]
                    d_dfl = time.perf_counter() - s
                    s = time.perf_counter()
                    sel = refdet.nms(cand, cand_sc, args.iou)
                    d_nms = time.perf_counter() - s
                    boxes, kept = cand[sel], cand_sc[sel]
                    n_cand += len(cand)
            else:
                s = time.perf_counter()
                keep, sc, _shapes = fastpost.candidate_keep(outs, args.conf)
                d_sco = time.perf_counter() - s
                if keep.size:
                    s = time.perf_counter()
                    xyxy = fastpost.boxes_xyxy_for(outs, args.imgsz, keep)
                    d_dfl = time.perf_counter() - s
                    s = time.perf_counter()
                    sel = fastpost.nms(np.ascontiguousarray(xyxy.T), sc, args.iou)
                    d_nms = time.perf_counter() - s
                    boxes, kept = np.ascontiguousarray(xyxy.T)[sel], sc[sel]
                    n_cand += keep.size
            n_kept += len(boxes)
            n_det += 1

        s = time.perf_counter()
        if boxes.shape[0]:
            nb = (boxes - np.array([dx, dy, dx, dy], np.float32)) / r
            xywh = np.empty((nb.shape[0], 4), np.float32)
            xywh[:, 0] = (nb[:, 0] + nb[:, 2]) * 0.5
            xywh[:, 1] = (nb[:, 1] + nb[:, 3]) * 0.5
            xywh[:, 2] = nb[:, 2] - nb[:, 0]
            xywh[:, 3] = nb[:, 3] - nb[:, 1]
        else:
            xywh = np.zeros((0, 4), np.float32)
        if args.dets == "array":
            dets = None
        else:
            dets = [{"xywh": xywh[i].copy(), "conf": float(kept[i])}
                    for i in range(xywh.shape[0])]
        d_bld = time.perf_counter() - s
        s = time.perf_counter()
        if args.dets == "array":
            tracker.update(xywh, kept)
        else:
            tracker.update(dets)
        d_trk = time.perf_counter() - s

        if args.dump_dir is not None:
            Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(args.dump_dir) / f"{fi + 1}.txt", "w") as fh_out:
                for k in range(len(tracker.xywh)):
                    if tracker.age[k] != 0:
                        continue
                    cx, cy, bw, bh = tracker.xywh[k]
                    fh_out.write(f"0 {cx / fw:.6f} {cy / fh:.6f} "
                                 f"{bw * args.box_scale / fw:.6f} "
                                 f"{bh * args.box_scale / fh:.6f} "
                                 f"{int(tracker.ids[k])} {float(tracker.conf[k]):.4f}\n")
        n_tracks = max(n_tracks, len(tracker.xywh))
        n_rows += len(xywh)
        if acc:
            t_pre += d_pre
            t_trk += d_trk
            t_bld += d_bld
            if fi % args.stride == 0:
                t_inf += d_inf
                t_sco += d_sco
                t_dfl += d_dfl
                t_nms += d_nms
            n_out += 1
    wall = time.perf_counter() - wall0

    nd = max(1, n_det)
    nf = max(1, n_out)
    print(f"  (timed over {n_out} frames after {args.warmup} warmup)")
    print(f"  preprocess     {t_pre * 1000 / nf:7.2f} ms/frame")
    print(f"  NPU inference  {t_inf * 1000 / nd:7.2f} ms/detection  ({n_det} detections)")
    print(f"    class scores {t_sco * 1000 / nd:7.2f} ms/detection")
    print(f"    DFL + bbox   {t_dfl * 1000 / nd:7.2f} ms/detection")
    print(f"    NMS          {t_nms * 1000 / nd:7.2f} ms/detection")
    print(f"  decode + NMS   {(t_sco + t_dfl + t_nms) * 1000 / nd:7.2f} ms/detection")
    print(f"  box convert    {t_bld * 1000 / nf:7.2f} ms/frame  "
          f"(detections/frame {n_rows / nf:.1f}, peak tracks {n_tracks})")
    print(f"  tracker        {t_trk * 1000 / nf:7.2f} ms/frame")
    per_det = (t_inf + t_sco + t_dfl + t_nms) / nd
    per_frame = (t_pre + t_bld + t_trk) / nf + per_det / args.stride
    print(f"  ------------------------------------------------")
    print(f"  detection frame {per_det * 1000:7.2f} ms  ->  {1 / per_det:5.1f} FPS detector")
    print(f"  output frame    {per_frame * 1000:7.2f} ms  ->  {1 / per_frame:5.1f} FPS tracked output")
    print(f"  wall clock      {wall:.2f} s for {len(frames)} frames "
          f"({len(frames) / wall:.1f} FPS end to end, warmup included)")
    if args.counts:
        print(f"  candidates {n_cand / nd:.1f}/detection, kept {n_kept / nd:.1f}/detection")
    rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
