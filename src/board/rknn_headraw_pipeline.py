#!/usr/bin/env python3
"""Board-side end-to-end pipeline timing: head-raw RKNN -> CPU decode -> tracker.

Why
---
rknn_headraw_detect.py measures the NPU forward pass only (57 ms on RK3568 for
the 448 head-raw INT8 model).  The deployed pipeline also has to letterbox, decode
the DFL on the CPU, run NMS and update the tracker, and none of that is in the
57 ms.  This script times every stage in the same process so the effective frame
rate is a measurement instead of an estimate.

The tracker here is a self-contained IoU + velocity tracker with the same
structure as run_tracking_deepocsort.Tracker (predict by velocity, high-conf pass
then low-conf pass, new tracks from high-conf unmatched), minus the appearance
embedding -- which was measured to contribute exactly 0.000 to dev MOTA.  It is
deliberately dependency-free (no torch, no scipy) so it runs on the board; the
assignment is greedy rather than Hungarian, so treat the MOTA implications as
indicative and the TIMING as the result.

    python3 rknn_headraw_pipeline.py --rknn dyhead30_448_headraw_int8.rknn \
        --imgsz 448 --stride 2 --img_dir val50 --limit 100

    python3 rknn_headraw_pipeline.py --rknn dyhead30_448_headraw_int8.rknn \
        --imgsz 448 --stride 2 --video /path/to/54.mp4 --limit 300
"""

import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np

from rknn_headraw_detect import boxes_for, class_scores_fast, letterbox, nms


def iou_xywh(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, xywh, conf):
        self.id = tid
        self.xywh = np.asarray(xywh, np.float32)
        self.vel = np.zeros(4, np.float32)
        self.conf = conf
        self.age = 0

    def predict(self):
        self.xywh = self.xywh + self.vel

    def update(self, xywh, conf):
        new = np.asarray(xywh, np.float32)
        self.vel = 0.8 * self.vel + 0.2 * (new - self.xywh)
        self.xywh = new
        self.conf = conf
        self.age = 0


def iou_matrix(A, B):
    """IoU between every (T,4) and (D,4) xywh box, vectorised.

    The first version of this used a Python double loop; at ~1000 tracks it cost
    550 ms per frame on the RK3568 and dominated the whole pipeline.  Broadcast
    numpy does the same work in well under a millisecond.
    """
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), np.float32)
    ax1, ay1 = A[:, 0] - A[:, 2] / 2, A[:, 1] - A[:, 3] / 2
    ax2, ay2 = A[:, 0] + A[:, 2] / 2, A[:, 1] + A[:, 3] / 2
    bx1, by1 = B[:, 0] - B[:, 2] / 2, B[:, 1] - B[:, 3] / 2
    bx2, by2 = B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2
    iw = np.maximum(0.0, np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]))
    ih = np.maximum(0.0, np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]))
    inter = iw * ih
    union = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def greedy_match(tracks, dets, thresh):
    """Greedy IoU assignment (board-friendly stand-in for the Hungarian match)."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    M = iou_matrix(np.stack([t.xywh for t in tracks]), np.stack([d["xywh"] for d in dets]))
    ii, jj = np.nonzero(M >= thresh)
    if ii.size == 0:
        return [], list(range(len(tracks))), list(range(len(dets)))
    order = np.argsort(-M[ii, jj])
    ii, jj = ii[order].tolist(), jj[order].tolist()
    ut, ud, matched = set(), set(), []
    for i, j in zip(ii, jj):
        if i in ut or j in ud:
            continue
        ut.add(i)
        ud.add(j)
        matched.append((i, j))
    return matched, [i for i in range(len(tracks)) if i not in ut], \
        [j for j in range(len(dets)) if j not in ud]


def optimal_match(tracks, dets, thresh):
    """Hungarian assignment -- what run_tracking_deepocsort.Tracker.match does.

    `greedy_match` above is a stand-in, and on video 82 frames 1-300 the
    substitution was measured to cost 0.048 MOTA: with the host's own detections,
    the greedy matcher gave 0.7081 against 0.7562 for the real tracker.  The
    signature was FP +64% with recall unchanged, which is what mis-assignment
    looks like -- in a frame holding ~30 sperm, taking the best IoU first and
    never reconsidering pairs the wrong tracks with the wrong detections.

    The matrix here is at most ~100x100, so the optimal assignment costs
    microseconds; there was never a speed reason for the stand-in.
    """
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    M = iou_matrix(np.stack([t.xywh for t in tracks]), np.stack([d["xywh"] for d in dets]))
    valid = M >= thresh
    if not valid.any():
        return [], list(range(len(tracks))), list(range(len(dets)))
    try:
        from scipy.optimize import linear_sum_assignment
        cost = np.where(valid, 1.0 - M, 1e6)
        rows, cols = linear_sum_assignment(cost)
        matched = [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]
    except Exception:                      # no scipy on the board: keep the fallback
        return greedy_match(tracks, dets, thresh)
    mt = {i for i, _ in matched}
    md = {j for _, j in matched}
    return matched, [i for i in range(len(tracks)) if i not in mt], \
        [j for j in range(len(dets)) if j not in md]


class Tracker:
    """Same association logic as run_tracking_deepocsort.Tracker, but the track
    state lives in arrays.

    The deployed tracker keeps a Python list of Track objects and does
    `for t in self.tracks: t.predict(); t.age += 1` every frame.  At ~100 live
    tracks that is ~300 numpy calls per frame before any matching happens, which
    measured ~10 ms on the RK3568 A55 -- comparable to the whole decode stage.
    Keeping xywh/vel/age as (T, 4) / (T,) arrays turns it into three vector ops.
    """

    def __init__(self, track_high, new_track, track_low, match_iou, low_iou, max_age):
        self.next_id = 1
        self.xywh = np.zeros((0, 4), np.float32)
        self.vel = np.zeros((0, 4), np.float32)
        self.age = np.zeros((0,), np.int32)
        self.ids = np.zeros((0,), np.int64)
        self.conf = np.zeros((0,), np.float32)
        self.hi, self.new_th, self.lo = track_high, new_track, track_low
        self.miou, self.liou, self.max_age = match_iou, low_iou, max_age

    @property
    def tracks(self):
        return self.xywh          # length only; used by the timing loop

    def __len__(self):
        return len(self.xywh)

    def _match(self, sub_idx, dets, thresh):
        """Optimal assignment between the selected tracks and dets.

        Was greedy (take the highest IoU pair, never reconsider); the host uses
        scipy's Hungarian solver.  Keeping the greedy path only as a fallback for
        a board without scipy -- see optimal_match() for why this matters.
        """
        if sub_idx.size == 0 or not dets:
            return [], list(range(sub_idx.size)), list(range(len(dets)))
        M = iou_matrix(self.xywh[sub_idx], np.stack([d["xywh"] for d in dets]))
        valid = M >= thresh
        if not valid.any():
            return [], list(range(sub_idx.size)), list(range(len(dets)))
        try:
            from scipy.optimize import linear_sum_assignment
            cost = np.where(valid, 1.0 - M, 1e6)
            rows, cols = linear_sum_assignment(cost)
            matched = [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]
        except Exception:
            ii, jj = np.nonzero(valid)
            order = np.argsort(-M[ii, jj])
            ii, jj = ii[order].tolist(), jj[order].tolist()
            ut, ud, matched = set(), set(), []
            for i, j in zip(ii, jj):
                if i in ut or j in ud:
                    continue
                ut.add(i)
                ud.add(j)
                matched.append((i, j))
        mt = {i for i, _ in matched}
        md = {j for _, j in matched}
        return matched, [i for i in range(sub_idx.size) if i not in mt], \
            [j for j in range(len(dets)) if j not in md]

    def _apply(self, rows, dets, cols):
        if not rows:
            return
        new = np.stack([dets[j]["xywh"] for j in cols])
        rows_arr = np.asarray(rows)
        old = self.xywh[rows_arr]
        self.vel[rows_arr] = 0.8 * self.vel[rows_arr] + 0.2 * (new - old)
        self.xywh[rows_arr] = new
        self.conf[rows_arr] = np.asarray([dets[j]["conf"] for j in cols], np.float32)
        self.age[rows_arr] = 0

    def update(self, dets):
        self.xywh = self.xywh + self.vel
        self.age += 1
        high = [d for d in dets if d["conf"] >= self.hi]
        low = [d for d in dets if self.lo <= d["conf"] < self.hi]

        all_idx = np.arange(len(self.xywh))
        matched, unr, unc = self._match(all_idx, high, self.miou)
        if matched:
            self._apply([m[0] for m in matched], high, [m[1] for m in matched])

        if unr and low:
            rem = all_idx[unr]
            matched2, _, _ = self._match(rem, low, self.liou)
            if matched2:
                self._apply([rem[m[0]] for m in matched2], low, [m[1] for m in matched2])

        born_xywh, born_conf = [], []
        for j in unc:
            if high[j]["conf"] >= self.new_th:
                born_xywh.append(high[j]["xywh"])
                born_conf.append(high[j]["conf"])
        if born_xywh:
            n = len(born_xywh)
            self.xywh = np.concatenate([self.xywh, np.stack(born_xywh)], 0)
            self.vel = np.concatenate([self.vel, np.zeros((n, 4), np.float32)], 0)
            self.age = np.concatenate([self.age, np.zeros(n, np.int32)])
            self.conf = np.concatenate([self.conf, np.asarray(born_conf, np.float32)])
            self.ids = np.concatenate([self.ids, np.arange(self.next_id, self.next_id + n)])
            self.next_id += n

        live = self.age <= self.max_age
        if not live.all():
            self.xywh, self.vel = self.xywh[live], self.vel[live]
            self.age, self.conf, self.ids = self.age[live], self.conf[live], self.ids[live]
        return self.xywh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rknn", required=True)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--stride", type=int, default=2, help="run the detector every k-th frame")
    ap.add_argument("--video", default=None)
    ap.add_argument("--img_dir", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--track-high", type=float, default=0.50)
    ap.add_argument("--new-track", type=float, default=0.50)
    ap.add_argument("--track-low", type=float, default=0.05)
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--low-match-iou", type=float, default=0.50)
    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--box-scale", type=float, default=1.0,
                    help="per-session output box scale (the deployed BOX_SCALE): multiplies ONLY "
                         "the reported width/height about the centre, exactly as "
                         "run_tracking_deepocsort.py does.  With the label-free rule the value "
                         "is kappa*(f_mean*f_peak*hf_ratio*lapvar)**0.25, kappa = 15.735 "
                         "(82:0.80 60:1.10 54:1.25 52:0.95).")
    ap.add_argument("--dump-dir", default=None,
                    help="write MOT-format labels (normalised cx cy w h id conf) so the board's "
                         "own detections can be scored against ground truth on the host.  This "
                         "is the only way to measure the INT8 model's MOTA: every MOTA this "
                         "project quotes so far comes from the FP32 model.")
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

    tracker = Tracker(args.track_high, args.new_track, args.track_low,
                      args.match_iou, args.low_match_iou, args.max_age)

    t_pre = t_inf = t_sco = t_dfl = t_nms = t_trk = 0.0
    n_det = n_out = 0
    n_tracks = 0
    n_dets = []
    print(f"\n{name}: {len(frames)} frames, detector stride {args.stride}")

    wall0 = time.perf_counter()
    # Reuse one canvas: np.full((448,448,3),114) allocates 600 KB every frame,
    # and the colour conversion is cheaper on the 448x448 crop than on the full
    # frame, so resize first and convert second.
    canvas = np.full((args.imgsz, args.imgsz, 3), 114, dtype=np.uint8)
    for fi, frame in enumerate(frames):
        acc = fi >= args.warmup          # warmup frames run but are not timed
        s = time.perf_counter()
        fh, fw = frame.shape[:2]
        r = args.imgsz / max(fh, fw)
        nh, nw = max(1, int(round(fh * r))), max(1, int(round(fw * r)))
        small = cv2.cvtColor(cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR),
                             cv2.COLOR_BGR2RGB)
        dx, dy = (args.imgsz - nw) // 2, (args.imgsz - nh) // 2
        canvas[:] = 114
        canvas[dy:dy + nh, dx:dx + nw] = small
        d_pre = time.perf_counter() - s

        dets = []
        d_inf = d_sco = d_dfl = d_nms = 0.0
        if fi % args.stride == 0:
            s = time.perf_counter()
            outs = rknn.inference(inputs=[canvas[None, ...]], data_format="nhwc")
            d_inf = time.perf_counter() - s
            s = time.perf_counter()
            scores, shapes = class_scores_fast(outs)
            keep = np.flatnonzero(scores > args.conf)
            d_sco = time.perf_counter() - s
            boxes = np.zeros((0, 4), np.float32)
            kept = np.zeros((0,), np.float32)
            if keep.size:
                s = time.perf_counter()
                cxcywh = boxes_for(outs, args.imgsz, keep)
                cx, cy, w, h = cxcywh
                cand = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
                cand_sc = scores[keep]
                d_dfl = time.perf_counter() - s
                s = time.perf_counter()
                sel = nms(cand, cand_sc, args.iou)
                d_nms = time.perf_counter() - s
                boxes, kept = cand[sel], cand_sc[sel]
            for b, sc in zip(boxes, kept):
                b = (b - np.array([dx, dy, dx, dy], np.float32)) / r
                dets.append({"xywh": np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                                               b[2] - b[0], b[3] - b[1]], np.float32),
                             "conf": float(sc)})

        s = time.perf_counter()
        tracker.update(dets)
        d_trk = time.perf_counter() - s
        if args.dump_dir is not None:
            Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(args.dump_dir) / f"{fi + 1}.txt", "w") as fh_out:
                sc = args.box_scale
                for k in range(len(tracker.xywh)):
                    # Match run_tracking_deepocsort.Tracker.update(), which returns a
                    # track only when `t.hits >= MIN_HITS and t.age == 0`.  Emitting
                    # every track with age <= max_age instead adds boxes for tracks
                    # that are merely coasting on velocity -- a box parked where the
                    # object used to be, which costs FP and buys no recall.
                    if tracker.age[k] != 0:
                        continue
                    cx, cy, bw, bh = tracker.xywh[k]
                    fh_out.write(f"0 {cx / fw:.6f} {cy / fh:.6f} "
                                 f"{bw * sc / fw:.6f} {bh * sc / fh:.6f} "
                                 f"{int(tracker.ids[k])} {float(tracker.conf[k]):.4f}\n")
        n_tracks = max(n_tracks, len(tracker.tracks))
        n_dets.append(len(dets))
        if acc:
            t_pre += d_pre
            t_trk += d_trk
            if fi % args.stride == 0:
                t_inf += d_inf
                t_sco += d_sco
                t_dfl += d_dfl
                t_nms += d_nms
                n_det += 1
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
    print(f"  tracker        {t_trk * 1000 / nf:7.2f} ms/frame")
    print(f"                 (peak live tracks {n_tracks}, "
          f"detections/frame max {max(n_dets) if n_dets else 0})")
    per_det = (t_inf + t_sco + t_dfl + t_nms) / nd
    per_frame = (t_pre + t_trk) / nf + per_det / args.stride
    print(f"  ------------------------------------------------")
    print(f"  detection frame {per_det * 1000:7.2f} ms  ->  {1 / per_det:5.1f} FPS detector")
    print(f"  output frame    {per_frame * 1000:7.2f} ms  ->  {1 / per_frame:5.1f} FPS tracked output")
    print(f"  wall clock      {wall:.2f} s for {len(frames)} frames "
          f"({len(frames) / wall:.1f} FPS end to end, warmup included)")
    rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
