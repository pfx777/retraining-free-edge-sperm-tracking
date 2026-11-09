#!/usr/bin/env python3
"""Can the per-session box-scale correction be predicted WITHOUT labels?

Why this exists
---------------
The per-session box-size calibration is the largest measured gain in this
project (+0.23 to +0.30 MOTA across four architectures), and its one
embarrassment is that it needs ground truth: the scalar is chosen on a small
annotated slice of the session.

Several label-free criteria are dead on arrival, and it is worth writing down
why before trying more:

  * matching the detected size distribution to the training distribution --
    the model's output is compressed toward its learned prior (all four videos
    emit a median of 16-18 input px), so matching returns s ~ 1;
  * any statistic of the detections alone -- videos 54 and 82 both emit a
    median detected side of ~16.2 input px but need opposite corrections
    (scale 1.30 vs 0.85).  Identical observable, opposite correction:
    detection-only statistics cannot decide;
  * temporal self-consistency of a track's box size -- invariant to a global
    scale by construction, so it cannot rank scales;
  * maximising the IoU between consecutive detections of one track -- degenerate,
    it grows monotonically to 1 as the scale goes to infinity.

What is NOT dead
----------------
Object size and background texture are produced by the same optics: the
magnification that makes a sperm head 13 px instead of 20 px scales everything
in the frame, including the debris.  So a *texture-scale* statistic of the raw
image is a label-free signal that is causally linked to the quantity the
calibration is trying to undo.

This script measures a few such cues per session and asks the one question that
decides the matter: do any of them separate the sessions in the same order as
the measured optimal scale?  In particular, can anything separate video 54
(needs 1.30) from video 82 (needs 0.85)?

It is a falsification test first and an estimator second: if the sessions that
need opposite corrections are indistinguishable on every cue, a label-free
calibration is impossible here and the annotated-slice protocol has to be
stated as the method.  If a cue separates them, that cue is the basis of a
label-free estimator that is calibrated once per rig.

    python analyze_session_scale_cues.py --vids 82,60,54,52 --frames 60 \
        --scales 82:0.85,60:1.30,54:1.30,52:0.95
    python analyze_session_scale_cues.py --self-test
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/code")


def image_cues(gray: np.ndarray) -> dict[str, float]:
    """Label-free texture/focus cues of a single greyscale frame."""
    g = gray.astype(np.float32)
    g -= g.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    # cycles per pixel, normalised so 0.5 is Nyquist in the shorter axis
    fx = (xx - cx) / w
    fy = (yy - cy) / h
    r = np.sqrt(fx**2 + fy**2)
    keep = r > 0
    rk, pk = r[keep], power[keep]
    if pk.sum() <= 0:
        return {"f_mean": 0.0, "f_peak": 0.0, "hf_ratio": 0.0, "lapvar": 0.0}
    f_mean = float((rk * pk).sum() / pk.sum())
    # radial profile, to find where the spectrum peaks (DC excluded)
    nb = 64
    idx = np.clip((rk / (rk.max() + 1e-9) * (nb - 1)).astype(np.int32), 0, nb - 1)
    prof = np.bincount(idx, weights=pk, minlength=nb)
    f_peak = float((prof.argmax() + 0.5) / nb * rk.max())
    # share of power above the median radius: a scale-free "how fine is it" cue
    hf_ratio = float(pk[rk > np.median(rk)].sum() / pk.sum())
    lap = np.abs(np.gradient(np.gradient(g, axis=0), axis=0) + np.gradient(np.gradient(g, axis=1), axis=1))
    return {"f_mean": f_mean, "f_peak": f_peak, "hf_ratio": hf_ratio, "lapvar": float(lap.mean())}


def self_test() -> int:
    """A coarse texture must read as a LOWER spatial frequency than a fine one."""
    rng = np.random.default_rng(0)
    def blur(a, sigma):
        k = int(4 * sigma) | 1
        x = np.arange(k) - k // 2
        gk = np.exp(-(x**2) / (2 * sigma**2))
        gk /= gk.sum()
        out = np.apply_along_axis(lambda m: np.convolve(m, gk, mode="same"), 0, a)
        return np.apply_along_axis(lambda m: np.convolve(m, gk, mode="same"), 1, out)

    fine = blur(rng.random((256, 256)), 1.0)
    coarse = blur(rng.random((256, 256)), 4.0)
    cf, cc = image_cues(fine * 255), image_cues(coarse * 255)
    assert cf["f_mean"] > cc["f_mean"], (cf, cc)
    assert cf["f_peak"] >= cc["f_peak"], (cf, cc)
    assert cf["lapvar"] > cc["lapvar"], (cf, cc)
    # the cues must be scale-free: rescaling intensity must not change them
    # (float32 arithmetic, so 1e-6 is the honest tolerance here)
    assert abs(image_cues(fine * 100)["f_mean"] - cf["f_mean"]) < 1e-6, (
        image_cues(fine * 100)["f_mean"], cf["f_mean"])
    flat = np.full((64, 64), 7.0)
    assert image_cues(flat)["f_mean"] == 0.0
    print("self-test OK: a coarse texture reads as lower spatial frequency and lower "
          "fine-detail energy than a fine one, and intensity scaling is neutral")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--frames", type=int, default=60, help="frames sampled evenly per video")
    ap.add_argument("--seed", type=int, default=None,
                    help="if set, draw the frames at random (this seed) rather than evenly.  "
                         "Re-running with several seeds measures how much the cues -- and hence "
                         "any predictor built on them -- move with nothing but the frame sample.")
    ap.add_argument("--cues-out", default=None,
                    help="write {\"<video>\": {cue: value}} so a predictor can be re-evaluated on "
                         "this frame sample instead of the one it was built on")
    ap.add_argument("--scales", default="82:0.85,60:1.30,54:1.30,52:0.95",
                    help="the MEASURED optimal scale per video, for the ordering test")
    ap.add_argument("--imgsz", type=int, default=448)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()

    import cv2
    sys.path.insert(0, str(ROOT))
    import run_tracking_deepocsort as rtd  # noqa: E402
    from sweep_box_curve import frame_size  # noqa: E402

    known = {}
    for item in args.scales.split(","):
        v, s = item.split(":")
        known[v.strip()] = float(s)

    rows = []
    for vid in [v.strip() for v in args.vids.split(",") if v.strip()]:
        video = rtd.SRC / vid / f"{vid}.mp4"
        gt_path = rtd.MOT_GT / f"{vid}.txt"
        if not video.exists():
            print(f"  video {vid}: no video at {video}")
            continue
        iw, ih = frame_size(video)
        letter = args.imgsz / max(ih, iw)
        gt_med = float("nan")
        if gt_path.exists():
            gt = rtd.parse_mot(str(gt_path))
            sides = [math.sqrt(max(b["bbox"][2] * iw * b["bbox"][3] * ih, 1e-6)) * letter
                     for boxes in gt.values() for b in boxes]
            gt_med = float(np.median(sides)) if sides else float("nan")

        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        n_take = min(args.frames, total)
        if args.seed is None:
            take = np.linspace(0, max(total - 1, 0), n_take).astype(int)
        else:
            take = np.sort(np.random.default_rng(args.seed).choice(total, size=n_take, replace=False))
        acc = []
        for fi in take:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            acc.append(image_cues(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        cap.release()
        if not acc:
            print(f"  video {vid}: could not read frames")
            continue
        row = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        row.update({"vid": vid, "gt_med": gt_med, "scale": known.get(vid, float("nan")),
                    "n": len(acc)})
        rows.append(row)
        print(f"  video {vid}: {row['n']} frames  GT median {gt_med:5.1f} px  "
              f"f_mean={row['f_mean']:.4f}  f_peak={row['f_peak']:.4f}  "
              f"hf_ratio={row['hf_ratio']:.3f}  lapvar={row['lapvar']:.1f}")

    if len(rows) < 3:
        print("need at least three sessions to test ordering")
        return 1

    if args.cues_out:
        import json
        Path(args.cues_out).write_text(json.dumps(
            {r["vid"]: {k: r[k] for k in ("f_mean", "f_peak", "hf_ratio", "lapvar")}
             for r in rows}, indent=2))
        print(f"\nwrote {args.cues_out}")

    print("\n" + "=" * 86)
    print("Can a label-free cue order the sessions like the measured optimal scale?")
    print(f"  {'video':<7}{'GT med px':>11}{'optimal s':>11}{'f_mean':>10}{'f_peak':>10}"
          f"{'hf_ratio':>10}{'lapvar':>10}")
    for r in sorted(rows, key=lambda x: x["scale"]):
        print(f"  {r['vid']:<7}{r['gt_med']:>11.1f}{r['scale']:>11.2f}"
              f"{r['f_mean']:>10.4f}{r['f_peak']:>10.4f}{r['hf_ratio']:>10.3f}{r['lapvar']:>10.1f}")
    print("  (rows sorted by the optimal scale; a usable cue must vary monotonically down these rows)")

    print("\nthe decisive pair: the two sessions that need OPPOSITE corrections")
    for pair in (("54", "82"), ("54", "60"), ("82", "52")):
        a = next((r for r in rows if r["vid"] == pair[0]), None)
        b = next((r for r in rows if r["vid"] == pair[1]), None)
        if not a or not b:
            continue
        print(f"  {pair[0]} (s={a['scale']:.2f}) vs {pair[1]} (s={b['scale']:.2f}):")
        for cue in ("gt_med", "f_mean", "f_peak", "hf_ratio", "lapvar"):
            va, vb = a[cue], b[cue]
            rel = abs(va - vb) / (abs(va) + abs(vb) + 1e-12) * 2
            print(f"     {cue:<9} {va:>10.4f} vs {vb:>10.4f}   relative gap {rel:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
