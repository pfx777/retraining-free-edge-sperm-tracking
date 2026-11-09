#!/usr/bin/env python3
"""Adaptive detection scheduling: spend the NPU budget where the tracker is
actually in trouble, instead of on every k-th frame.

Why
---
The delivered FPS/MOTA curve is a *fixed-k* curve: the detector runs on frames
1, 1+k, 1+2k, ...  Every published number shows the same shape -- MOTA falls
with k, and the fall is concentrated in FN, because a track that is not refreshed
keeps coasting on its velocity.  But the loss is not spread evenly over frames:
the frames that hurt are the ones where the tracker is already uncertain (new
objects appearing, many coasting tracks, poor association IoU).

An adaptive schedule can therefore buy the same frame rate for less MOTA: keep
the *average* detection rate equal to 1/k, but choose which frames to spend it
on.  The signal is computed from state the tracker already has, so it is causal
and needs no labels.

What is measured
----------------
Replaying the board's own stride-1 detection dump for each session (nothing is
re-detected, no board time is used):

  fixed-k        the published curve, reproduced in this harness
  random         same budget, frames picked at random  <-- the control
  urgency        detect when the tracker's instability signal, grown by the
                 number of frames since the last detection, crosses a threshold

If `urgency` does not beat `random` at the same detection rate, the signal is not
carrying information -- the same placebo logic used for the TPC loss-weighting
experiment.

    python3 adaptive_sched_eval.py --dets-dir _dumps_v8dets --out logs/adaptive_sched.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score_det_dump import (  # noqa: E402
    load_board_tracker, mota, interpolate, parse_mot,
)

# per-session deployed box scale (label-free kappa rule, host side)
BOX_SCALE = {"82": 0.90, "60": 1.40, "54": 1.60, "52": 0.95}


def load_dets(path: Path, fw: float, fh: float):
    """detections.txt -> {frame: [(level, score, (cx, cy, w, h) in pixels)]}"""
    per_frame: dict[int, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        q = line.split()
        if len(q) < 6:
            continue
        fr = int(float(q[0]))
        cx, cy, w, h, s = (float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5]))
        lv = int(float(q[6])) if len(q) > 6 else 0
        per_frame[fr].append((lv, s, (cx * fw, cy * fh, w * fw, h * fh)))
    return per_frame


class Replay:
    """One session, one scheduling policy -> MOT tracks (id -> frame -> box)."""

    def __init__(self, tracker_cls, args, per_frame, decision, fw, fh, emit_age=0):
        self.tr = tracker_cls(args.track_high, args.new_track, args.track_low,
                              args.match_iou, args.low_match_iou, args.max_age)
        self.args = args
        self.per_frame = per_frame
        self.decision = decision          # callable(state) -> bool
        self.fw, self.fh = fw, fh
        self.tracks: dict[int, dict[int, tuple]] = defaultdict(dict)
        self.n_det_frames = 0
        self.n_frames = 0
        self.emit_age = emit_age
        self.state = dict(instab=0.0, since=999, n_tracks=0, mean_iou=1.0)
        self._prev_ids = set()

    def _predict_iou(self, dets):
        """Mean best-IoU between the tracker's *predicted* boxes and the incoming
        high-confidence detections -- a free, causal association-quality signal."""
        if len(self.tr.xywh) == 0 or not dets:
            return 0.0
        A = self.tr.xywh + self.tr.vel
        B = np.stack([d["xywh"] for d in dets])
        ax1, ay1 = A[:, 0] - A[:, 2] / 2, A[:, 1] - A[:, 3] / 2
        ax2, ay2 = A[:, 0] + A[:, 2] / 2, A[:, 1] + A[:, 3] / 2
        bx1, by1 = B[:, 0] - B[:, 2] / 2, B[:, 1] - B[:, 3] / 2
        bx2, by2 = B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2
        iw = np.maximum(0.0, np.minimum(ax2[:, None], bx2[None, :])
                        - np.maximum(ax1[:, None], bx1[None, :]))
        ih = np.maximum(0.0, np.minimum(ay2[:, None], by2[None, :])
                        - np.maximum(ay1[:, None], by1[None, :]))
        inter = iw * ih
        union = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None, :] - inter
        M = inter / np.maximum(union, 1e-9)
        return float(np.mean(M.max(axis=1)))

    def _assoc_stats(self, mean_iou=None):
        """State right after the last detection, from the tracker's own arrays."""
        tr = self.tr
        n = len(tr.xywh)
        if n == 0:
            return dict(instab=1.0, n_tracks=0, mean_iou=0.0, new_tracks=0, coasting=0)
        age = np.asarray(tr.age)
        fresh = int((age == 0).sum())
        coasting = n - fresh
        ids = set(int(i) for i in tr.ids)
        new = len(ids - self._prev_ids)
        self._prev_ids = ids
        instab = (new + 0.5 * coasting) / max(n, 1)
        return dict(instab=float(instab), n_tracks=n,
                    mean_iou=float(1.0 if mean_iou is None else mean_iou),
                    new_tracks=int(new), coasting=int(coasting))

    def run(self, lo, hi, vid, box_scale):
        for fr in range(lo, hi + 1):
            self.n_frames += 1
            rows = self.per_frame.get(fr, [])
            st = dict(self.state)
            st["since"] = self.state["since"] + 1
            do_det = self.decision(st, fr) if rows else False
            dets = []
            miou = None
            if do_det:
                self.n_det_frames += 1
                if getattr(self.args, "signal_iou", False):
                    high = [{"xywh": np.array([cx, cy, w * box_scale, h * box_scale],
                                              np.float32), "conf": s}
                            for lv, s, (cx, cy, w, h) in rows if s >= self.args.track_high]
                    miou = self._predict_iou(high)
                for lv, s, (cx, cy, w, h) in rows:
                    dets.append({"xywh": np.array([cx, cy, w * box_scale, h * box_scale],
                                                  np.float32), "conf": s})
            self.tr.update(dets)
            if do_det:
                stats = self._assoc_stats(miou)
                stats["since"] = 0
                self.state = stats
            else:
                self.state = st
                self.state["since"] = st["since"]
            for k in range(len(self.tr.xywh)):
                if self.tr.age[k] > self.emit_age:
                    continue
                cx, cy, w, h = self.tr.xywh[k]
                self.tracks[int(self.tr.ids[k])][fr] = (cx / self.fw, cy / self.fh,
                                                        w / self.fw, h / self.fh)
        return self.tracks, self.n_det_frames, self.n_frames


def make_policy(args, kind, param, seed=0):
    """Returns (decision(state, frame) -> bool, label)."""
    if kind == "fixedk":
        k = int(param[0])
        return (lambda st, fr, k=k: (fr - 1) % k == 0), f"fixedk{k}"
    if kind == "random":
        p = float(param[0])
        rng = np.random.default_rng(seed)
        return (lambda st, fr, p=p, rng=rng: bool(rng.random() < p)), f"random{p:.3f}"
    if kind == "urgency":
        tau = float(param[0])
        mode = param[1] if len(param) > 1 else "instab"
        kmax = int(param[2]) if len(param) > 2 else 12

        def dec(st, fr, tau=tau, mode=mode, kmax=kmax):
            if st["since"] >= kmax:
                return True
            if mode == "instab":
                u = st["instab"]
            elif mode == "born":
                u = 1.0 if st["new_tracks"] > 0 else 0.0
            elif mode == "coast":
                u = st["coasting"] / max(st["n_tracks"], 1)
            else:
                u = 1.0 - st["mean_iou"]
            return (u * (1.0 + st["since"])) >= tau
        return dec, f"urgency{tau:.2f}_{mode}_k{kmax}"
    if kind == "periodic_hard":     # denser detection on a schedule, no signal
        k = int(param[0])
        return (lambda st, fr, k=k: (fr - 1) % k == 0), f"fixedk{k}"
    raise SystemExit(f"unknown policy {kind}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets-dir", required=True,
                    help="directory holding one subdir per session, each with detections.txt")
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--gt-dir", default=str(HERE / "datasets" / "visem_tracking" / "mot_gt_c0"),
                    help="class-0-only ground truth (the unified convention: the detector "
                         "is class-0-only and the MOTA matcher ignores classes)")
    ap.add_argument("--board-script", default=str(HERE / "rknn_headraw_pipeline.py"))
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=100000)
    ap.add_argument("--gap", type=int, default=14)
    ap.add_argument("--fw", type=float, default=640.0)
    ap.add_argument("--fh", type=float, default=480.0)
    ap.add_argument("--track-high", type=float, default=0.60)
    ap.add_argument("--new-track", type=float, default=0.60)
    ap.add_argument("--track-low", type=float, default=0.05)
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--low-match-iou", type=float, default=0.70)
    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra-policies", action="store_true",
                    help="also sweep the born/coast/iou signal variants")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--fixedk", default="1,2,3,4",
                    help="strides for the constant-rate baseline, e.g. 1,2,3,4,6,8")
    ap.add_argument("--only-fixedk", action="store_true")
    ap.add_argument("--signal-iou", action="store_true",
                    help="also compute the association-quality signal (a few extra "
                         "numpy ops per detected frame)")
    args = ap.parse_args()

    Tracker = load_board_tracker(Path(args.board_script))
    vids = [v for v in args.vids.split(",") if v]
    data = {}
    for v in vids:
        p = Path(args.dets_dir) / f"dets_{v}" / "detections.txt"
        if not p.exists():
            p = Path(args.dets_dir) / v / "detections.txt"
        per_frame = load_dets(p, args.fw, args.fh)
        hi = min(args.hi, max(per_frame))
        gtp = Path(args.gt_dir) / f"{v}.txt"
        gt = parse_mot(gtp)
        gt = {f: b for f, b in gt.items() if args.lo <= f <= hi}
        data[v] = (per_frame, gt, hi)
        print(f"{v}: frames {args.lo}..{hi}  det rows {sum(len(x) for x in per_frame.values())}")

    policies = []
    for k in [int(x) for x in args.fixedk.split(",") if x]:
        policies.append(("fixedk", (k,)))
    if not args.only_fixedk:
        for tau in (1.0, 1.5, 2.0, 3.0):
            policies.append(("urgency", (tau, "instab", 12)))
    if args.extra_policies and not args.only_fixedk:
        for tau in (1.0, 2.0):
            policies.append(("urgency", (tau, "born", 12)))
            policies.append(("urgency", (tau, "coast", 12)))
        if args.signal_iou:
            for tau in (1.0, 2.0):
                policies.append(("urgency", (tau, "iou", 12)))

    rows = []
    for kind, param in policies:
        per_vid = {}
        rates = {}
        for v in vids:
            per_frame, gt, hi = data[v]
            dec, label = make_policy(args, kind, param)
            rep = Replay(Tracker, args, per_frame, dec, args.fw, args.fh)
            tracks, nd, nf = rep.run(args.lo, hi, v, BOX_SCALE.get(v, 1.0))
            m = mota(gt, interpolate(tracks, args.gap))
            per_vid[v] = m["MOTA"]
            rates[v] = nd / max(nf, 1)
        mean = float(np.mean(list(per_vid.values())))
        rate = float(np.mean(list(rates.values())))
        rows.append(dict(policy=label, kind=kind, rate=rate,
                         **{f"mota_{v}": per_vid[v] for v in vids},
                         mota_mean=mean))
        print(f"{label:24s} rate {rate:.3f}  "
              + "  ".join(f"{v} {per_vid[v]:.4f}" for v in vids)
              + f"   mean {mean:.4f}")

    # control: random schedule, budget matched to each urgency run
    print("\ncontrol: random schedule at the matched rate")
    matched = [r for r in rows if r["kind"] == "urgency"]
    for r in matched:
        for seed in range(args.seeds):
            per_vid, rates = {}, {}
            for v in vids:
                per_frame, gt, hi = data[v]
                dec, _lbl = make_policy(args, "random", (r["rate"],), seed=seed)
                rep = Replay(Tracker, args, per_frame, dec, args.fw, args.fh)
                tracks, nd, nf = rep.run(args.lo, hi, v, BOX_SCALE.get(v, 1.0))
                m = mota(gt, interpolate(tracks, args.gap))
                per_vid[v] = m["MOTA"]
                rates[v] = nd / max(nf, 1)
            mean = float(np.mean(list(per_vid.values())))
            rr = dict(policy=f"random@rate({r['policy']})_s{seed}", kind="random",
                      rate=float(np.mean(list(rates.values()))),
                      **{f"mota_{v}": per_vid[v] for v in vids}, mota_mean=mean)
            rows.append(rr)
            print(f"{rr['policy']:28s} rate {rr['rate']:.3f}  "
                  + "  ".join(f"{v} {per_vid[v]:.4f}" for v in vids)
                  + f"   mean {mean:.4f}")

    # the fixed-k curve, re-read as a function of rate, so every scheduled run can
    # be compared against a constant-rate schedule with the same budget
    fk = sorted([(r["rate"], r["mota_mean"]) for r in rows if r["kind"] == "fixedk"],
                key=lambda t: t[0])
    xs = [t[0] for t in fk]
    ys = [t[1] for t in fk]

    def fixedk_at(rate):
        return float(np.interp(rate, xs, ys))

    print("\nmatched-budget comparison (fixed-k interpolated at the same rate)")
    hdr = f"{'policy':28s} {'rate':>6s} {'MOTA':>8s} {'fixed-k':>8s} {'delta':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda t: t["rate"]):
        base = fixedk_at(r["rate"])
        print(f"{r['policy']:28s} {r['rate']:6.3f} {r['mota_mean']:8.4f} {base:8.4f} "
              f"{r['mota_mean'] - base:+8.4f}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = ["policy", "kind", "rate", "mota_mean"] + [f"mota_{v}" for v in vids]
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in keys})
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
