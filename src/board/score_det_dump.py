#!/usr/bin/env python3
"""Run the board's own tracker on a detection dump, so post-processing choices
can be scored without another board run.

`rknn_dump_dets.py` writes every candidate the board produced, with its level
and its raw score.  This script puts those back through the SAME tracker class
the board pipeline uses (sliced out of `rknn_headraw_pipeline.py`, not
reimplemented) and scores the result with the project's MOTA.

    --box-scale-lN   multiplies w,h of level-N detections (the board's --box-scale,
                     but per level)
    --score-aN/-bN   remaps level-N scores in logit space: sigmoid(a*logit(s)+b),
                     the label-free head recalibration

Sanity check against a known number (host detections, board tracker, video 82
frames 1..300, box-scale 0.80): MOTA 0.7544.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

MOT_GT = Path("/root/code/datasets/visem_tracking/mot_gt")


def load_board_tracker(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def iou_xywh"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("def main()"))
    ns: dict = {"np": np}
    exec(compile("".join(lines[start:end]), str(path), "exec"), ns)
    return ns["Tracker"]


def parse_mot(path: Path) -> dict[int, list]:
    out: dict[int, list] = {}
    for line in path.read_text().splitlines():
        p = line.replace(",", " ").split()
        if len(p) >= 6:
            out.setdefault(int(float(p[0])), []).append(
                (int(float(p[1])), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return out


def iou(a, b) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax1 + a[2], bx1 + b[2]), min(ay1 + a[3], by1 + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def interpolate(tracks, gap: int) -> dict[int, list]:
    out: dict[int, list] = defaultdict(list)
    for tid, frames in tracks.items():
        for f, b in frames.items():
            out[f].append((tid, b))
        fs = sorted(frames)
        for a, b2 in zip(fs, fs[1:]):
            h = b2 - a - 1
            if 0 < h <= gap:
                for m in range(a + 1, b2):
                    al = (m - a) / (b2 - a)
                    out[m].append((tid, tuple(l + al * (r - l)
                                              for l, r in zip(frames[a], frames[b2]))))
    return out


def mota(gt: dict[int, list], pred: dict[int, list]) -> dict:
    """Identical to mota_selfcontained.mota: greedy IoU>=0.5 matching per frame,
    ID switches counted against each GT's previously matched prediction id."""
    fp = fn = idsw = tp = ngt = 0
    prev: dict[int, int] = {}
    for fr in sorted(gt):
        g = gt[fr]
        ngt += len(g)
        p = pred.get(fr, [])
        M = np.zeros((len(g), len(p)), dtype=np.float64)
        for i, (gid, *gb) in enumerate(g):
            for j, (tid, pb) in enumerate(p):
                M[i, j] = iou(gb, pb)
        used_g, used_p = set(), set()
        pairs = sorted(((M[i, j], i, j) for i in range(len(g)) for j in range(len(p))
                        if M[i, j] >= 0.5), reverse=True)
        for _, i, j in pairs:
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            tp += 1
            gid, tid = g[i][0], p[j][0]
            if gid in prev and prev[gid] != tid:
                idsw += 1
            prev[gid] = tid
        fn += len(g) - len(used_g)
        fp += len(p) - len(used_p)
    return {"MOTA": 1.0 - (fp + fn + idsw) / max(ngt, 1), "FP": fp, "FN": fn,
            "IDSW": idsw, "TP": tp, "GT": ngt}


def score_logit_remap(s, a, b):
    if a == 1.0 and b == 0.0:
        return s
    q = np.clip(s, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    return 1.0 / (1.0 + np.exp(-np.clip(a * z + b, -60.0, 60.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vid", required=True)
    ap.add_argument("--dets", required=True, help="detections.txt from rknn_dump_dets.py")
    ap.add_argument("--gt-dir", default=None,
                    help="ground-truth directory (default: the server's "
                         "/root/code/datasets/visem_tracking/mot_gt)")
    ap.add_argument("--board-script", default="/root/code/rknn_headraw_pipeline.py")
    ap.add_argument("--box-scale-l0", type=float, default=1.0)
    ap.add_argument("--box-scale-l1", type=float, default=1.0)
    ap.add_argument("--box-scale-l2", type=float, default=1.0)
    ap.add_argument("--box-scale-h0", type=float, default=None,
                    help="separate height factor for level 0; the tensor "
                         "measurement says the conversion error is anisotropic "
                         "(w 1.23, h 1.30), so w and h may need different "
                         "corrections.  Defaults to the width factor.")
    ap.add_argument("--box-scale-h1", type=float, default=None)
    ap.add_argument("--box-scale-h2", type=float, default=None)
    ap.add_argument("--score-a0", type=float, default=1.0)
    ap.add_argument("--score-b0", type=float, default=0.0)
    ap.add_argument("--score-a1", type=float, default=1.0)
    ap.add_argument("--score-b1", type=float, default=0.0)
    ap.add_argument("--score-a2", type=float, default=1.0)
    ap.add_argument("--score-b2", type=float, default=0.0)
    ap.add_argument("--score-map-json", default=None,
                    help="piecewise score map produced by make_score_map.py "
                         "(board score -> reference score, per level).  Applied "
                         "after the affine remap, and it replaces the affine one "
                         "for the levels it covers.")
    ap.add_argument("--track-high", type=float, default=0.60)
    ap.add_argument("--new-track", type=float, default=0.60)
    ap.add_argument("--track-low", type=float, default=0.05)
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--low-match-iou", type=float, default=0.70)
    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--gap", type=int, default=14)
    ap.add_argument("--fw", type=float, default=640.0)
    ap.add_argument("--fh", type=float, default=480.0)
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=100000)
    ap.add_argument("--det-stride", type=int, default=1,
                    help="emulate the board running its detector every k-th frame: "
                         "detections are fed on frames where (frame-1) %% k == 0 "
                         "and the tracker coasts on the rest.  The dump has every "
                         "frame, so this gives the board's own MOTA(k) curve with "
                         "no extra board time.")
    ap.add_argument("--dup-suppress", type=float, default=0.0,
                    help="track-level duplicate suppression: when two tracks sit "
                         "within this many mean box widths of each other on a frame "
                         "AND their sizes agree within --dup-size-ratio, drop the "
                         "lower-scored one for its whole lifetime.  Motivated by the "
                         "measured failure mode: the FP are persistent duplicates of "
                         "already-tracked targets, and GT targets are 5-7 box widths "
                         "apart, so 'same object' is decidable without labels.")
    ap.add_argument("--dup-size-ratio", type=float, default=2.0)
    ap.add_argument("--score-gain", type=float, default=0.0,
                    help="optional per-level score gain applied in logit space "
                         "before the tracker (0 = off)")
    args = ap.parse_args()

    scale = {0: args.box_scale_l0, 1: args.box_scale_l1, 2: args.box_scale_l2}
    hscale = {0: args.box_scale_h0 if args.box_scale_h0 is not None else args.box_scale_l0,
              1: args.box_scale_h1 if args.box_scale_h1 is not None else args.box_scale_l1,
              2: args.box_scale_h2 if args.box_scale_h2 is not None else args.box_scale_l2}
    remap = {0: (args.score_a0, args.score_b0), 1: (args.score_a1, args.score_b1),
             2: (args.score_a2, args.score_b2)}

    tables: dict[int, tuple] = {}
    if args.score_map_json:
        import json

        with open(args.score_map_json) as fh:
            blob = json.load(fh)
        for entry in blob["levels"]:
            lv = int(entry["level"])
            if entry.get("identity"):
                continue
            tables[lv] = (np.asarray(entry["board"], np.float64),
                          np.asarray(entry["ref"], np.float64))
        print(f"score map {args.score_map_json}: levels {sorted(tables)}")

    per_frame: dict[int, list] = defaultdict(list)
    levels = defaultdict(int)
    for line in Path(args.dets).read_text().splitlines():
        q = line.split()
        if len(q) < 6:
            continue
        fr = int(float(q[0]))
        cx, cy, w, h, s = (float(q[1]), float(q[2]), float(q[3]), float(q[4]),
                           float(q[5]))
        lv = int(float(q[6])) if len(q) > 6 else 0
        levels[lv] += 1
        per_frame[fr].append((lv, s, (cx * args.fw, cy * args.fh,
                                      w * args.fw, h * args.fh)))

    Tracker = load_board_tracker(Path(args.board_script))
    tracker = Tracker(args.track_high, args.new_track, args.track_low,
                      args.match_iou, args.low_match_iou, args.max_age)

    frames_seen = sorted(per_frame)
    hi = min(args.hi, max(frames_seen) if frames_seen else 0)
    tracks: dict[int, dict[int, tuple]] = defaultdict(dict)
    n_dets = 0
    for fr in range(args.lo, hi + 1):
        dets = []
        rows = per_frame.get(fr, []) if (fr - args.lo) % args.det_stride == 0 else []
        for lv, s, box in rows:
            s2 = float(score_logit_remap(np.array([s]), *remap.get(lv, (1.0, 0.0)))[0])
            if lv in tables:
                xp, yp = tables[lv]
                s2 = float(np.interp(s2, xp, yp))
            sc = scale.get(lv, 1.0)
            sh = hscale.get(lv, 1.0)
            cx, cy, w, h = box
            dets.append({"xywh": np.array([cx, cy, w * sc, h * sh], np.float32),
                         "conf": s2})
        n_dets += len(dets)
        tracker.update(dets)
        for k in range(len(tracker.xywh)):
            if tracker.age[k] != 0:
                continue
            cx, cy, w, h = tracker.xywh[k]
            tracks[int(tracker.ids[k])][fr] = (cx / args.fw, cy / args.fh,
                                               w / args.fw, h / args.fh)

    if args.dup_suppress > 0.0 and tracks:
        # Geometric duplicate suppression, label-free: rank tracks by how often
        # they were actually updated (a real target is matched on most frames),
        # then drop any track that sits on top of a better-ranked one.
        support = {i: len(v) for i, v in tracks.items()}
        order = sorted(tracks, key=lambda i: -support[i])
        drop = set()
        frames = set()
        for v in tracks.values():
            frames.update(v)
        for fr in sorted(frames):
            present = [i for i in order if fr in tracks[i] and i not in drop]
            for a in range(len(present)):
                ia = present[a]
                if ia in drop:
                    continue
                ba = tracks[ia][fr]
                for b in range(a + 1, len(present)):
                    ib = present[b]
                    if ib in drop:
                        continue
                    bb = tracks[ib][fr]
                    wmean = 0.5 * (ba[2] + bb[2])
                    dist = ((ba[0] - bb[0]) ** 2 + (ba[1] - bb[1]) ** 2) ** 0.5
                    ratio = max(ba[2] / max(bb[2], 1e-9), bb[2] / max(ba[2], 1e-9))
                    if dist < args.dup_suppress * wmean and ratio <= args.dup_size_ratio:
                        drop.add(ib)
        for i in drop:
            tracks.pop(i, None)
        print(f"  duplicate suppression: {len(drop)}/{len(support)} tracks dropped "
              f"(alpha={args.dup_suppress}, ratio<={args.dup_size_ratio})")

    gt = parse_mot((Path(args.gt_dir) if args.gt_dir else MOT_GT) / f"{args.vid}.txt")
    gt = {f: b for f, b in gt.items() if args.lo <= f <= hi}
    m = mota(gt, interpolate(tracks, args.gap))
    print(f"video {args.vid} frames {args.lo}..{hi}  detections {n_dets} "
          f"(levels {dict(levels)})  tracks {len(tracks)}")
    print(f"  MOTA {m['MOTA']:.4f}  FP {m['FP']}  FN {m['FN']}  IDSW {m['IDSW']} "
          f"TP {m['TP']}  GT {m['GT']}")
    print(f"  recall {m['TP'] / max(m['TP'] + m['FN'], 1):.4f}  "
          f"precision {m['TP'] / max(m['TP'] + m['FP'], 1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
