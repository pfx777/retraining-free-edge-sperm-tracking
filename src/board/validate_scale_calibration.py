#!/usr/bin/env python3
"""Validate a per-video BOX_SCALE calibration protocol.

Why this is needed
------------------
sweep_per_video_scale.py picks the best BOX_SCALE per video *on the same frames
it then scores*, which is test-set tuning and not a usable claim.  This script
implements the protocol a fixed microscope rig would actually use:

    calibrate the scale on the first N frames of each video,
    then score MOTA on the remaining frames with that scale frozen.

Two reference points are reported alongside it, so the cost of the protocol is
visible rather than implied:

  * global BOX_SCALE=1.00 on the eval range      -- no calibration at all
  * best scale chosen on the eval range itself   -- oracle upper bound

The honest number to quote is the middle one (calib-selected on eval).

Measured with --calib-mode head (the naive "first N frames" protocol) this was:

    calib-frac     global 1.00   calib-selected   oracle
       0.05            0.530          0.579        0.661
       0.10            0.521          0.567        0.656
       0.20            0.504          0.612        0.646
       0.30            0.488          0.619        0.632

i.e. the protocol recovered only a third of the available gain, and *lengthening*
the head clip helped.  The reason is visible per video: the per-video optimum is
not stationary.  Video 60's first 10% prefers BOX_SCALE 0.95 (0.776) while its
remaining 90% prefers 1.20 (0.624) -- head calibration picks the scale that is
worst on the part being scored, dropping video 60 from 0.624 to 0.385.  Video 82
is the opposite: head and tail agree on 0.85 and the protocol costs nothing.

So the failure is the *sampling* (calibrating on one segment and scoring the
other), not the clip length.  --calib-mode even / random spread the calibration
frames over the whole video and should recover most of the head-vs-oracle gap.

Predictions are the per-frame label files already written by
run_tracking_deepocsort.py / run_tracking_interp_sweep.py, so no tracking is
re-run -- only the frame range used for scoring changes.

    python validate_scale_calibration.py --run-name ablation_lkctx30 --imgsz 640 \
        --scales 0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20

    # different calibration length, and a different output-dir layout
    python validate_scale_calibration.py --run-name ablation_dyhead30 --imgsz 448 \
        --scales 0.80,0.85,...,1.20 --calib-frac 0.1 --calib-mode even
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

TRACK_ROOT = "/root/code/runs/visem_tracking"
DEFAULT_TEMPLATE = TRACK_ROOT + "/pvscale_{run}_{imgsz}_{vid}_{scale}"
# The first per-video scale sweep was launched as an inline shell loop and wrote
# to this layout instead; both are probed so a protocol run can mix them.
LEGACY_TEMPLATE = TRACK_ROOT + "/scalecal_{vid}_{scale}"

# Local smoke-test convenience: prefer a bundled ultralytics source tree if the
# package is not installed.  No effect on the server, where it is installed.
for _extra in (Path(__file__).resolve().parent / "ultralytics-8.3.163", Path(__file__).resolve().parent):
    if _extra.is_dir() and str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


def find_labels(templates: list[str], run: str, imgsz: str, vid: str, scale: str) -> Path | None:
    """First path holding label files for this (vid, scale).

    The tracker writes to ``{OUT_DIR}/{vid}/labels/*.txt``, so a template that
    names the run output dir has to be descended into; a template that already
    names the labels dir is accepted as-is.

    Two scale spellings are probed for every template, because the sweeps did not
    agree on one: sweep_per_video_scale.py strips the dot
    (``pvscale_<run>_640_54_120``) while the earlier inline shell loop kept it
    (``scalecal_54_1.20``).
    """
    for tmpl in templates:
        for spelling in (scale, scale.replace(".", "")):
            root = Path(tmpl.format(run=run, imgsz=imgsz, vid=vid, scale=spelling))
            for cand in (root / vid / "labels", root):
                if cand.is_dir() and any(cand.glob("*.txt")):
                    return cand
    return None


def load_pred(labels_dir: Path) -> dict[int, list[dict]]:
    """Read the per-frame label files written by the tracker."""
    pred: dict[int, list[dict]] = defaultdict(list)
    if not labels_dir.is_dir():
        return pred
    for f in sorted(labels_dir.glob("*.txt")):
        frame = int(f.stem.rsplit("_", 1)[-1])
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 6:
                continue
            pred[frame].append({
                "id": int(float(p[5])),
                "bbox": [float(p[1]), float(p[2]), float(p[3]), float(p[4])],
            })
    return pred


def load_tracks(labels_dir: Path) -> dict[int, dict[int, tuple]]:
    """Same labels, regrouped as {track_id: {frame: (cx, cy, w, h)}}."""
    tracks: dict[int, dict[int, tuple]] = {}
    if not labels_dir.is_dir():
        return tracks
    for f in sorted(labels_dir.glob("*.txt")):
        frame = int(f.stem.rsplit("_", 1)[-1])
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 6:
                continue
            tracks.setdefault(int(float(p[5])), {})[frame] = (
                float(p[1]), float(p[2]), float(p[3]), float(p[4]))
    return tracks


def interp_to_pred(interp: dict[int, list]) -> dict[int, list[dict]]:
    return {f: [{"id": tid, "bbox": list(box)} for tid, box in items]
            for f, items in interp.items()}


def slice_frames(d: dict[int, list[dict]], lo: int, hi: int) -> dict[int, list[dict]]:
    return {f: v for f, v in d.items() if lo <= f < hi}


def subset_frames(d: dict[int, list[dict]], keep: set[int]) -> dict[int, list[dict]]:
    return {f: v for f, v in d.items() if f in keep}


def pick_calib_frames(all_frames: list[int], n_calib: int, mode: str, seed: int) -> set[int]:
    """Frames used for calibration.

    ``head`` is the naive protocol (first N frames) and is the one that fails:
    video 60's first 10% prefers BOX_SCALE 0.95 while the remaining 90% prefers
    1.20, so calibrating on the head picks the scale that is worst on the part
    that is scored.  ``even`` and ``random`` spread the calibration frames over
    the whole video so the scale estimate is not tied to one segment.
    """
    n_calib = max(1, min(n_calib, len(all_frames) - 1))
    if mode == "head":
        return set(all_frames[:n_calib])
    if mode == "even":
        step = len(all_frames) / n_calib
        return {all_frames[min(len(all_frames) - 1, int(i * step))] for i in range(n_calib)}
    if mode == "random":
        return set(random.Random(seed).sample(all_frames, n_calib))
    raise ValueError(f"unknown calib mode {mode}")


def self_test() -> int:
    """Build a fake tracker output tree and check the lookup + loader.

    Guards the two bugs that made the first server run fail: labels live at
    ``{OUT_DIR}/{vid}/labels``, and a scale with no directory must yield "missing"
    rather than a KeyError.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmpl = str(Path(td) / "pvscale_{run}_{imgsz}_{vid}_{scale}")
        lab = Path(td) / "pvscale_demo_640_54_120" / "54" / "labels"
        lab.mkdir(parents=True)
        (lab / "54_3.txt").write_text("0 0.5 0.5 0.1 0.1 7 1.0\n", encoding="utf-8")

        got = find_labels([tmpl], "demo", "640", "54", "1.20")
        assert got == lab, f"find_labels returned {got}, expected {lab}"
        pred = load_pred(lab)
        assert list(pred) == [3], pred
        assert pred[3][0]["id"] == 7 and pred[3][0]["bbox"] == [0.5, 0.5, 0.1, 0.1], pred

        missing = find_labels([tmpl], "demo", "640", "54", "0.85")
        assert missing is None, f"missing scale resolved to {missing}"

        # the earlier inline sweep kept the dot in the directory name
        dotted = Path(td) / "scalecal_54_0.85" / "54" / "labels"
        dotted.mkdir(parents=True)
        (dotted / "54_1.txt").write_text("0 0.4 0.4 0.1 0.1 9 1.0\n", encoding="utf-8")
        got_dotted = find_labels([str(Path(td) / "scalecal_{vid}_{scale}")], "demo", "640", "54", "0.85")
        assert got_dotted == dotted, f"dot-preserving lookup returned {got_dotted}, expected {dotted}"

        # sparse raw labels (what DET_STRIDE>1 produces) must be fillable before
        # slicing, otherwise every skipped frame counts as a false negative
        sparse = Path(td) / "holdout52_0.80" / "52" / "labels"
        sparse.mkdir(parents=True)
        (sparse / "52_1.txt").write_text("0 0.10 0.10 0.02 0.02 7 1.0\n", encoding="utf-8")
        (sparse / "52_9.txt").write_text("0 0.18 0.10 0.02 0.02 7 1.0\n", encoding="utf-8")
        dense = interp_to_pred(__import__("run_tracking_interp_sweep").interpolate_tracks(
            load_tracks(sparse), 14))
        assert sorted(dense) == list(range(1, 10)), sorted(dense)
        assert abs(dense[5][0]["bbox"][0] - 0.14) < 1e-6, dense[5]
        assert len(load_pred(sparse)) == 2, "raw labels really are sparse"

        print("self-test OK: find_labels descends into {vid}/labels, load_pred round-trips, "
              "missing scale returns None, both scale spellings resolve, "
              "sparse raw labels interpolate to dense")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="check the label lookup and loader, then exit")
    ap.add_argument("--run-name", default=None, help="run directory name (not needed with --self-test)")
    ap.add_argument("--imgsz", default="640")
    ap.add_argument("--scales", default="0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20")
    # Full validation split by default.  Video 52 was held out during
    # development; that phase is over, so every reported number now covers all
    # four val videos and is directly comparable.
    ap.add_argument("--vids", default="82,60,54,52")
    ap.add_argument("--baseline-scale", default="1.00")
    ap.add_argument("--calib-frac", type=float, default=0.1,
                    help="fraction of each video's frames used for calibration")
    ap.add_argument("--calib-frames", type=int, default=None,
                    help="absolute calibration length; overrides --calib-frac")
    ap.add_argument("--calib-mode", choices=("head", "even", "random"), default="head",
                    help="which frames to calibrate on.  'head' reproduces the naive "
                         "first-N-frames protocol; 'even'/'random' spread the sample over "
                         "the whole video, which matters because the per-video optimum is "
                         "not stationary (video 60's head prefers scale 0.95, its tail 1.20)")
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--interp-gap", type=int, default=0,
                    help="fill gaps in the raw labels before slicing, using the same "
                         "interpolate_tracks the tracker pipeline uses.  REQUIRED when the "
                         "grid was produced with DET_STRIDE>1: the tracker emits boxes only "
                         "on frames where it matched a detection, so the raw labels are "
                         "sparse and scoring them directly reports mass false negatives.")
    ap.add_argument("--raw-template", default=DEFAULT_TEMPLATE,
                    help="per-(video,scale) label dir; {run} {imgsz} {vid} {scale}. "
                         "The legacy scalecal_{vid}_{scale} layout is probed as a fallback.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.run_name:
        ap.error("--run-name is required")

    sys.path.insert(0, "/root/code")
    import run_tracking_deepocsort as rtd  # noqa: E402  (provides MOT_GT/parse_mot/mot_metrics)

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    vids = [v.strip() for v in args.vids.split(",") if v.strip()]

    rows: list[dict[str, object]] = []
    print(f"run={args.run_name} imgsz={args.imgsz} calib-mode={args.calib_mode} "
          f"calib={'frames=%d' % args.calib_frames if args.calib_frames else 'frac=%.2f' % args.calib_frac}")

    calib_motas: list[float] = []
    eval_motas: list[float] = []
    base_motas: list[float] = []
    oracle_motas: list[float] = []

    for vid in vids:
        gt_path = rtd.MOT_GT / f"{vid}.txt"
        if not gt_path.exists():
            print(f"  video {vid}: no GT at {gt_path}")
            continue
        gt_all = rtd.parse_mot(str(gt_path))
        gt_frames = sorted(gt_all)
        if not gt_frames:
            print(f"  video {vid}: empty GT")
            continue
        f0, f1 = gt_frames[0], gt_frames[-1] + 1
        n_calib = args.calib_frames or max(1, int(round(len(gt_frames) * args.calib_frac)))
        calib_set = pick_calib_frames(gt_frames, n_calib, args.calib_mode, args.calib_seed)
        eval_set = set(gt_frames) - calib_set
        gt_calib = subset_frames(gt_all, calib_set)
        gt_eval = subset_frames(gt_all, eval_set)

        preds = {}
        used_templates: dict[str, str] = {}
        for sc in scales:
            d = find_labels([args.raw_template, LEGACY_TEMPLATE], args.run_name, args.imgsz, vid, sc)
            if d is not None:
                if args.interp_gap > 0:
                    from run_tracking_interp_sweep import interpolate_tracks

                    preds[sc] = interp_to_pred(interpolate_tracks(load_tracks(d), args.interp_gap))
                else:
                    preds[sc] = load_pred(d)
                used_templates[sc] = str(d)
        missing = [sc for sc in scales if not preds.get(sc)]
        if missing:
            print(f"  video {vid}: WARNING no labels for scales {missing}")
            print(f"    probed {args.raw_template.format(run=args.run_name, imgsz=args.imgsz, vid=vid, scale=scales[0].replace('.', ''))}")
            print(f"    and    {LEGACY_TEMPLATE.format(vid=vid, scale=scales[0].replace('.', ''))}")
        elif used_templates:
            uniq = sorted(set(used_templates.values()))
            if len(uniq) > 1:
                print(f"  video {vid}: labels came from {len(uniq)} different directories "
                      f"(e.g. {uniq[0]} ... {uniq[-1]})")

        calib_curve = {}
        eval_curve = {}
        for sc in scales:
            if not preds.get(sc):
                continue
            calib_curve[sc] = rtd.mot_metrics(gt_calib, subset_frames(preds[sc], calib_set))
            eval_curve[sc] = rtd.mot_metrics(gt_eval, subset_frames(preds[sc], eval_set))
        if not eval_curve:
            print(f"  video {vid}: nothing to score")
            continue

        # protocol: pick the scale on the calibration range only
        chosen = max(calib_curve, key=lambda s: calib_curve[s]["MOTA"])
        oracle = max(eval_curve, key=lambda s: eval_curve[s]["MOTA"])
        base = args.baseline_scale if args.baseline_scale in eval_curve else None

        m_eval = eval_curve[chosen]
        calib_motas.append(calib_curve[chosen]["MOTA"])
        eval_motas.append(m_eval["MOTA"])
        if base:
            base_motas.append(eval_curve[base]["MOTA"])
        oracle_motas.append(eval_curve[oracle]["MOTA"])

        print(f"\n  video {vid}  frames {f0}..{f1 - 1}  "
              f"(calib {len(calib_set)} frames via '{args.calib_mode}', "
              f"eval {len(eval_set)} frames, GT {sum(len(v) for v in gt_eval.values())})")
        print("    scale   calib-MOTA   eval-MOTA")
        for sc in scales:
            if sc not in eval_curve:
                continue
            mark = ""
            if sc == chosen:
                mark += "  <- chosen on calib"
            if sc == oracle:
                mark += "  <- best on eval (oracle)"
            print(f"    {sc:<6}  {calib_curve[sc]['MOTA']:>9.3f}   {eval_curve[sc]['MOTA']:>9.3f}{mark}")
        print(f"    -> chosen scale {chosen}: eval MOTA {m_eval['MOTA']:.3f} "
              f"(FP={m_eval['FP']} FN={m_eval['FN']} IDSW={m_eval['IDSW']})")
        if base:
            print(f"    -> global {args.baseline_scale}: eval MOTA {eval_curve[base]['MOTA']:.3f}")
        print(f"    -> oracle on eval: scale {oracle}, MOTA {eval_curve[oracle]['MOTA']:.3f}")

        rows.append({
            "video": vid,
            "calib_scale": chosen, "calib_mota": round(calib_curve[chosen]["MOTA"], 4),
            "eval_mota": round(m_eval["MOTA"], 4),
            "global_scale": args.baseline_scale,
            "global_eval_mota": round(eval_curve[base]["MOTA"], 4) if base else "",
            "oracle_scale": oracle, "oracle_eval_mota": round(eval_curve[oracle]["MOTA"], 4),
            "eval_FP": m_eval["FP"], "eval_FN": m_eval["FN"], "eval_IDSW": m_eval["IDSW"],
        })

    print("\n" + "=" * 74)
    print(f"SUMMARY  {args.run_name} @ imgsz={args.imgsz}")
    if eval_motas:
        print(f"  calibrated on first {args.calib_frames or f'{args.calib_frac:.0%}'}"
              f" ({args.calib_mode}), scored on the rest:")
        if base_motas:
            print(f"    global {args.baseline_scale} (no calibration)  dev mean = "
                  f"{sum(base_motas) / len(base_motas):.3f}")
        print(f"    calib-selected                   dev mean = {sum(eval_motas) / len(eval_motas):.3f}")
        print(f"    oracle on eval (upper bound)     dev mean = {sum(oracle_motas) / len(oracle_motas):.3f}")
    if rows:
        out = Path(args.out) if args.out else Path(TRACK_ROOT) / f"scalecalib_{args.run_name}_{args.imgsz}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
