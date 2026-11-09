#!/usr/bin/env python3
"""Gated track interpolation sweep for the VISEM DeepOCSORT tracker.

This reuses raw labels already produced by ``run_tracking_deepocsort.py`` and
searches over (gap, max center step per frame, max box-size ratio).  The gate
is intended to retain most FN recovery from interpolation while rejecting
large jumps that are more likely to create false positives.
"""

import argparse
from pathlib import Path

from run_tracking_interp_sweep import (
    evaluate_videos,
    interpolate_tracks,
    load_tracks,
    write_tracks,
)


def parse_float_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(None if item.lower() == "inf" else float(item))
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaps", default="8,10,12")
    parser.add_argument("--max-step-list", default="0.03,0.05,0.10,inf")
    parser.add_argument("--size-ratio-list", default="1.5,2.0,3.0,inf")
    parser.add_argument(
        "--raw-dir",
        default="/root/code/runs/visem_tracking/deepocsort_interp_raw",
    )
    parser.add_argument(
        "--out-root",
        default="/root/code/runs/visem_tracking/deepocsort_interp_gated",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import run_tracking_deepocsort as rtd

    gaps = [int(x) for x in args.gaps.split(",") if x.strip()]
    step_list = parse_float_list(args.max_step_list)
    ratio_list = parse_float_list(args.size_ratio_list)
    vids = [v for v in rtd.VAL_VIDS if v.strip()]

    print("\n=== Gated track interpolation sweep ===")
    per_video_by_combo = {}

    for gap in gaps:
        for max_step in step_list:
            for max_size_ratio in ratio_list:
                combo = (gap, max_step, max_size_ratio)
                out_labels_root = (
                    Path(args.out_root)
                    / f"gap{gap}"
                    / f"step{max_step if max_step is not None else 'inf'}"
                    / f"ratio{max_size_ratio if max_size_ratio is not None else 'inf'}"
                )
                for vid in vids:
                    raw_labels = Path(args.raw_dir) / vid / "labels"
                    tracks = load_tracks(raw_labels)
                    interpolated = interpolate_tracks(
                        tracks,
                        gap,
                        max_step_per_frame=max_step,
                        max_size_ratio=max_size_ratio,
                    )
                    write_tracks(interpolated, out_labels_root / vid / "labels")

                tot, motas, per_video = evaluate_videos(rtd, vids, out_labels_root)
                per_video_by_combo[combo] = per_video
                mean_mota = sum(motas) / max(len(motas), 1)
                aggregate = 1 - (tot["FP"] + tot["FN"] + tot["IDSW"]) / max(tot["GT"], 1)
                print(
                    f"gap={gap} step={max_step if max_step is not None else 'inf'} "
                    f"ratio={max_size_ratio if max_size_ratio is not None else 'inf'}: "
                    f"Mean MOTA={mean_mota:.3f} Aggregate MOTA={aggregate:.3f} "
                    f"FP={tot['FP']} FN={tot['FN']} IDSW={tot['IDSW']}"
                )

    if not per_video_by_combo:
        return

    print("\n=== Per-video best gated combo ===")
    best_by_video = {}
    for vid in vids:
        choices = []
        for combo, metrics in per_video_by_combo.items():
            if vid in metrics:
                choices.append((combo, metrics[vid]))
        if not choices:
            continue
        best_combo, best_metric = max(choices, key=lambda item: item[1]["MOTA"])
        best_by_video[vid] = (best_combo, best_metric)
        gap, max_step, max_size_ratio = best_combo
        print(
            f"  {vid}: gap={gap} step={max_step if max_step is not None else 'inf'} "
            f"ratio={max_size_ratio if max_size_ratio is not None else 'inf'} "
            f"MOTA={best_metric['MOTA']:.3f} FP={best_metric['FP']} "
            f"FN={best_metric['FN']} IDSW={best_metric['IDSW']}"
        )

    motas = [metric["MOTA"] for _, metric in best_by_video.values()]
    tot = {"FP": 0, "FN": 0, "IDSW": 0, "GT": 0}
    for _, metric in best_by_video.values():
        for key in tot:
            tot[key] += metric[key]
    print(f"Per-video gated best Mean MOTA: {sum(motas) / len(motas):.3f}")
    print(
        f"Per-video gated best Aggregate MOTA: "
        f"{1 - (tot['FP'] + tot['FN'] + tot['IDSW']) / max(tot['GT'], 1):.3f}"
    )


if __name__ == "__main__":
    main()
