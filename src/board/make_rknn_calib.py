#!/usr/bin/env python3
"""Build a self-contained RKNN INT8 calibration package.

Why not just prepare_rknn_dataset.py
------------------------------------
That script writes absolute paths for whatever machine it runs on, so a
dataset.txt made on the server is useless on the VM where rknn-toolkit runs.
It also reads visem_merged rather than the visem_official split the models are
now trained on, and it does not check for leakage.

This script produces a folder that survives the transfer:

    <out-dir>/
        images/               N copied JPEGs
        dataset.txt           absolute paths for *this* machine (works here)
        make_dataset_txt.py   re-writes dataset.txt with the paths of whatever
                              machine it is run on -- run it once after copying
        SOURCE.txt            where the frames came from and the leak check

Safeguards
----------
* frames whose video id is in --protect (default 54,60,82,52) are excluded, so
  the calibration set can never contain a dev or holdout frame.  Video 52 in
  particular is the sealed holdout and must not appear anywhere.
* sampling is evenly spaced over the sorted frame list rather than taking the
  first N, so the set spans videos and time instead of one video's opening.

    python make_rknn_calib.py --n 200 --out-dir runs/deploy/rknn_calib
    tar czf runs/deploy/rknn_calib.tgz -C runs/deploy rknn_calib

    # on the VM
    tar xzf rknn_calib.tgz && cd rknn_calib && python make_dataset_txt.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Ordered by preference: the split the current models were trained on first.
SRC_CANDIDATES_TRAIN = (
    "/root/code/datasets/visem_official/train/images",
    "/root/code/datasets/visem_official/images/train",
    "/root/code/datasets/visem_official/train",
    "/root/code/datasets/visem_merged/train/images",
    "/root/code/datasets/visem_merged/images/train",
    "/root/code/datasets/visem_merged/train",
)
SRC_CANDIDATES_VAL = (
    "/root/code/datasets/visem_official/val/images",
    "/root/code/datasets/visem_merged/val/images",
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

REGEN_SCRIPT = '''#!/usr/bin/env python3
"""Re-write dataset.txt with absolute paths for the machine this runs on."""
from pathlib import Path

here = Path(__file__).resolve().parent
imgs = sorted((here / "images").glob("*"))
imgs = [p for p in imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
out = here / "dataset.txt"
out.write_text("\\n".join(str(p.resolve()) for p in imgs) + "\\n", encoding="utf-8")
print(f"wrote {out} with {len(imgs)} absolute paths")
print("sample:", str(imgs[0].resolve()) if imgs else "(none)")
'''


def video_id(name: str) -> str:
    m = re.match(r"(\d+)[_-]", Path(name).name)
    return m.group(1) if m else "?"


def pick_even(paths: list[Path], n: int) -> list[Path]:
    """Evenly spaced sample instead of the first N."""
    if n >= len(paths):
        return list(paths)
    return [paths[round(i * (len(paths) - 1) / (n - 1))] for i in range(n)]


def self_test() -> int:
    paths = [Path(f"{i}_frame_{j}.jpg") for i in (11, 12, 13) for j in range(10)]
    assert video_id("54_frame_1200.jpg") == "54"
    assert video_id("82-33.png") == "82"
    assert video_id("weird.jpg") == "?"

    sampled = pick_even(paths, 6)
    assert len(sampled) == 6, len(sampled)
    assert sampled[0] == paths[0] and sampled[-1] == paths[-1]
    ids = {video_id(p.name) for p in sampled}
    assert len(ids) >= 2, f"sample collapsed onto one video: {ids}"
    assert pick_even(paths, 100) == paths, "should return everything when n >= len"
    print("self-test OK: video-id parsing and even sampling both behave")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--out-dir", default="/root/code/runs/deploy/rknn_calib")
    ap.add_argument("--src", default=None, help="explicit source image directory")
    ap.add_argument("--protect", default="54,60,82,52",
                    help="video ids that must never appear (dev + sealed holdout)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    protect = {v.strip() for v in args.protect.split(",") if v.strip()}

    candidates = (args.src,) if args.src else (
        SRC_CANDIDATES_TRAIN if args.split == "train" else SRC_CANDIDATES_VAL)
    src = next((Path(c) for c in candidates if c and Path(c).is_dir()
                and any(Path(c).glob("*"))), None)
    if src is None:
        print("no source image directory found.  Tried:")
        for c in candidates:
            print(f"  {c}")
        print("\npass --src <dir> to point at one explicitly")
        return 1

    all_imgs = sorted(p for p in src.iterdir()
                      if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    if not all_imgs:
        print(f"{src} contains no images")
        return 1

    kept = [p for p in all_imgs if video_id(p.name) not in protect]
    dropped = len(all_imgs) - len(kept)
    print(f"source : {src}")
    print(f"found  : {len(all_imgs)} images, dropped {dropped} from protected videos {sorted(protect)}")
    if not kept:
        print("ERROR: nothing left after the leak filter; check --protect or --src")
        return 1

    sample = pick_even(kept, args.n)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_dir.mkdir(parents=True)
    for p in sample:
        shutil.copy2(p, img_dir / p.name)

    by_video: dict[str, int] = {}
    for p in sample:
        by_video[video_id(p.name)] = by_video.get(video_id(p.name), 0) + 1

    (out_dir / "dataset.txt").write_text(
        "\n".join(str((img_dir / p.name).resolve()) for p in sample) + "\n", encoding="utf-8")
    (out_dir / "make_dataset_txt.py").write_text(REGEN_SCRIPT, encoding="utf-8")
    (out_dir / "SOURCE.txt").write_text(
        f"source dir : {src}\nsplit      : {args.split}\n"
        f"total found: {len(all_imgs)}\ndropped    : {dropped} (protected videos {sorted(protect)})\n"
        f"sampled    : {len(sample)}\n"
        f"per video  : " + ", ".join(f"{k}:{v}" for k, v in sorted(by_video.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)) + "\n"
        f"leak check : " + ("CLEAN -- no protected video id present"
                            if not (set(by_video) & protect) else
                            "CONTAMINATED: " + str(sorted(set(by_video) & protect))) + "\n",
        encoding="utf-8")

    total_mb = sum(p.stat().st_size for p in sample) / 1e6
    print(f"sampled: {len(sample)} images  ({total_mb:.1f} MB)")
    print(f"per video: " + ", ".join(f"{k}:{v}" for k, v in sorted(by_video.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)))
    print(f"leak   : " + ("CLEAN (no protected video id)" if not (set(by_video) & protect)
                          else f"CONTAMINATED {sorted(set(by_video) & protect)}"))
    print(f"\nwrote {out_dir}/")
    print(f"  images/             {len(sample)} frames")
    print(f"  dataset.txt         absolute paths for this machine")
    print(f"  make_dataset_txt.py re-writes dataset.txt for the machine it runs on")
    print(f"  SOURCE.txt          provenance + leak check")
    print(f"\ntransfer:  tar czf {out_dir}.tgz -C {out_dir.parent} {out_dir.name}")
    print(f"on the VM: tar xzf {out_dir.name}.tgz && cd {out_dir.name} && python make_dataset_txt.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
