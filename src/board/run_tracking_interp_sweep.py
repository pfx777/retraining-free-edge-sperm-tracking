#!/usr/bin/env python3
"""Track-gap interpolation sweep for the VISEM DeepOCSORT tracker.

The raw DeepOCSORT labels only contain boxes on frames where a track was
matched.  When a track is briefly lost and then recovered with the same ID,
the gap is currently counted as FN.  This script:

  1. optionally runs ``run_tracking_deepocsort.main`` to generate raw labels;
  2. reads those labels per track ID;
  3. linearly interpolates boxes for gaps up to several frames;
  4. evaluates MOTA for every requested gap length in one pass.

It is intended to be run with the same environment variables as
``run_tracking_deepocsort.py`` (MODEL, REID_MODEL, VAL_VIDS, per-video
thresholds, etc.).
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path


def parse_frame_from_name(name):
    """Extract the frame index from a label filename like ``82_123.txt``."""
    stem = Path(name).stem
    return int(stem.rsplit("_", 1)[-1])


def load_tracks(labels_dir):
    """Return ``{track_id: {frame: (cx, cy, w, h)}}`` in normalized coords."""
    tracks = defaultdict(dict)
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        return tracks
    for label_file in sorted(labels_dir.glob("*.txt")):
        frame = parse_frame_from_name(label_file.name)
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 6:
                continue
            track_id = int(float(parts[5]))
            box = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            tracks[track_id][frame] = box
    return tracks


def interpolate_tracks(tracks, max_gap, max_step_per_frame=None, max_size_ratio=None):
    """Linearly fill missing frames for each track ID up to ``max_gap``.

    If ``max_step_per_frame`` is set, a gap is skipped when the normalized
    center displacement per frame exceeds that value.  If ``max_size_ratio``
    is set, a gap is skipped when either box side changes by more than that
    ratio between the two endpoint detections.
    """
    out = defaultdict(list)

    for track_id, frames in tracks.items():
        for frame, box in frames.items():
            out[frame].append((track_id, box))

        ordered = sorted(frames)
        for idx in range(len(ordered) - 1):
            left, right = ordered[idx], ordered[idx + 1]
            gap = right - left - 1
            if gap <= 0 or gap > max_gap:
                continue
            left_box = frames[left]
            right_box = frames[right]
            if max_step_per_frame is not None:
                step = max(
                    abs(right_box[0] - left_box[0]),
                    abs(right_box[1] - left_box[1]),
                ) / gap
                if step > max_step_per_frame:
                    continue
            if max_size_ratio is not None:
                bad_size = False
                for dim in (2, 3):
                    ratio = max(left_box[dim], right_box[dim]) / (min(left_box[dim], right_box[dim]) + 1e-6)
                    if ratio > max_size_ratio:
                        bad_size = True
                        break
                if bad_size:
                    continue
            for mid in range(left + 1, right):
                alpha = (mid - left) / (right - left)
                mid_box = tuple(l + alpha * (r - l) for l, r in zip(left_box, right_box))
                out[mid].append((track_id, mid_box))

    return out


def write_tracks(out, labels_dir):
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for frame, items in out.items():
        with (labels_dir / f"{frame}.txt").open("w") as f:
            for track_id, box in sorted(items, key=lambda x: x[0]):
                cx, cy, w, h = box
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {track_id} 1.0\n")


def evaluate_videos(rtd, vids, labels_root):
    tot = {"FP": 0, "FN": 0, "IDSW": 0, "GT": 0, "TP": 0}
    motas = []
    per_video = {}
    for vid in vids:
        gt_path = rtd.MOT_GT / f"{vid}.txt"
        labels_dir = Path(labels_root) / vid / "labels"
        if not gt_path.exists() or not labels_dir.exists():
            continue
        gt = rtd.parse_mot(str(gt_path))
        pred = defaultdict(list)
        for label_file in sorted(labels_dir.glob("*.txt")):
            frame = parse_frame_from_name(label_file.name)
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if len(parts) < 6:
                    continue
                pred[frame].append({
                    "id": int(float(parts[5])),
                    "bbox": [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                })
        metric = rtd.mot_metrics(gt, pred)
        for key in tot:
            tot[key] += metric[key]
        motas.append(metric["MOTA"])
        per_video[vid] = metric
    return tot, motas, per_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaps", default="1,2,3,4,5")
    parser.add_argument(
        "--raw-dir",
        default=os.environ.get("OUT_DIR", "/root/code/runs/visem_tracking/deepocsort_interp_raw"),
    )
    parser.add_argument(
        "--out-root",
        default="/root/code/runs/visem_tracking/deepocsort_interp",
    )
    parser.add_argument("--skip-run", action="store_true", help="reuse existing raw labels")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["OUT_DIR"] = str(args.raw_dir)

    import run_tracking_deepocsort as rtd

    if not args.skip_run:
        rtd.main()

    gaps = [int(x) for x in args.gaps.split(",") if x.strip()]
    vids = [v for v in rtd.VAL_VIDS if v.strip()]

    print("\n=== Track interpolation sweep ===")
    per_video_by_gap = {}
    for gap in gaps:
        out_labels_root = Path(args.out_root) / f"gap{gap}"
        for vid in vids:
            raw_labels = Path(args.raw_dir) / vid / "labels"
            tracks = load_tracks(raw_labels)
            interpolated = interpolate_tracks(tracks, gap)
            write_tracks(interpolated, out_labels_root / vid / "labels")

        tot, motas, per_video = evaluate_videos(rtd, vids, out_labels_root)
        per_video_by_gap[gap] = per_video
        mean_mota = sum(motas) / max(len(motas), 1)
        aggregate = 1 - (tot["FP"] + tot["FN"] + tot["IDSW"]) / max(tot["GT"], 1)
        print(
            f"gap={gap}: Mean MOTA={mean_mota:.3f} "
            f"Aggregate MOTA={aggregate:.3f} "
            f"FP={tot['FP']} FN={tot['FN']} IDSW={tot['IDSW']} GT={tot['GT']}"
        )

    if not per_video_by_gap:
        return

    print("\n=== Per-video best gap ===")
    best_by_video = {}
    for vid in vids:
        choices = []
        for gap, metrics in per_video_by_gap.items():
            if vid in metrics:
                choices.append((gap, metrics[vid]))
        if not choices:
            continue
        best_gap, best_metric = max(choices, key=lambda item: item[1]["MOTA"])
        best_by_video[vid] = (best_gap, best_metric)
        print(
            f"  {vid}: gap={best_gap} MOTA={best_metric['MOTA']:.3f} "
            f"FP={best_metric['FP']} FN={best_metric['FN']} IDSW={best_metric['IDSW']}"
        )

    if best_by_video:
        motas = [metric["MOTA"] for _, metric in best_by_video.values()]
        tot = {"FP": 0, "FN": 0, "IDSW": 0, "GT": 0}
        for _, metric in best_by_video.values():
            for key in tot:
                tot[key] += metric[key]
        print(f"Per-video best Mean MOTA: {sum(motas) / len(motas):.3f}")
        print(
            f"Per-video best Aggregate MOTA: "
            f"{1 - (tot['FP'] + tot['FN'] + tot['IDSW']) / max(tot['GT'], 1):.3f}"
        )


if __name__ == "__main__":
    main()
