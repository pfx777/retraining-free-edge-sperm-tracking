#!/usr/bin/env python3
"""
prepare_rknn_dataset.py — 生成 RKNN 校准集 dataset.txt
从 val 图片里挑 N 张, 每行一个绝对路径。
RKNN build(do_quantization=True) 需要这个文件做 INT8 激活校准。
"""
import glob, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200, help="校准图片数量")
parser.add_argument("--split", choices=["train", "val"], default="train", help="校准集来源，默认 train，避免测试集泄漏")
parser.add_argument("--out", default="/root/code/runs/deploy/dataset.txt")
args = parser.parse_args()

if args.split == "val":
    img_dirs = [
        "/root/code/datasets/visem_merged/val/images",
        "/root/code/datasets/visem_merged/images/val",
        "/root/code/datasets/visem_merged/val",
    ]
else:
    img_dirs = [
        "/root/code/datasets/visem_merged/train/images",
        "/root/code/datasets/visem_merged/images/train",
        "/root/code/datasets/visem_merged/train",
    ]
img_paths = []
for d in img_dirs:
    if Path(d).exists():
        img_paths = sorted(glob.glob(f"{d}/*.jpg") + glob.glob(f"{d}/*.png") + glob.glob(f"{d}/*.jpeg"))
        if img_paths:
            break
if not img_paths:
    raise SystemExit(f"No images found in {img_dirs}")

img_paths = img_paths[:args.n]
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(img_paths) + "\n", encoding="utf-8")
print(f"Wrote {len(img_paths)} images to {out}")
print("Sample lines:")
for line in img_paths[:3]:
    print(f"  {line}")
