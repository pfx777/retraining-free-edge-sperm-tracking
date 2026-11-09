#!/usr/bin/env python3
"""Evaluate board RKNN detections against YOLO-format GT labels.

Input CSV format produced by rknn_lite_eval.py:
    filename,conf,x1,y1,x2,y2

Outputs one-class mAP50 and mAP50-95.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def compute_ap(preds, gt_boxes_by_image, iou_thr):
    total_gt = sum(len(v) for v in gt_boxes_by_image.values())
    if total_gt == 0 or not preds:
        return 0.0, 0.0, 0.0

    preds = sorted(preds, key=lambda x: -x[0])
    matched = {img: [False] * len(boxes) for img, boxes in gt_boxes_by_image.items()}
    tp = np.zeros(len(preds), dtype=np.float32)
    fp = np.zeros(len(preds), dtype=np.float32)

    for i, (conf, img, box) in enumerate(preds):
        gt_boxes = gt_boxes_by_image.get(img, [])
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if matched[img][j]:
                continue
            cur = iou(box, gt)
            if cur > best_iou:
                best_iou = cur
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = 1.0
            matched[img][best_j] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    rec = cum_tp / total_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    precision = float(cum_tp[-1] / max(cum_tp[-1] + cum_fp[-1], 1e-9))
    recall = float(cum_tp[-1] / total_gt)
    return ap, precision, recall


def load_gt(label_dir, img_dir, wanted_names):
    gt_boxes_by_image = {}
    for name in sorted(wanted_names):
        img_path = Path(img_dir) / name
        if not img_path.exists():
            continue
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = parts[:5]
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append((x1, y1, x2, y2))
        gt_boxes_by_image[name] = boxes
    return gt_boxes_by_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dets", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold for reported Precision/Recall")
    args = parser.parse_args()

    preds = []
    names = set()
    with open(args.dets, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            name = row[0]
            names.add(name)
            conf = float(row[1])
            x1, y1, x2, y2 = map(float, row[2:6])
            preds.append((conf, name, (x1, y1, x2, y2)))

    gt = load_gt(args.labels, args.img_dir, names)
    preds = [p for p in preds if p[1] in gt]

    total_gt = sum(len(v) for v in gt.values())
    total_preds = len(preds)

    aps = {}
    for thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        ap, _, _ = compute_ap(preds, gt, thr)
        aps[f"ap_{int(thr*100):02d}"] = ap

    map50 = aps["ap_50"]
    map50_95 = float(np.mean(list(aps.values())))

    conf_preds = [p for p in preds if p[0] >= args.conf]
    _, ap50_prec, ap50_rec = compute_ap(conf_preds, gt, 0.5)

    print(f"GT boxes: {total_gt}")
    print(f"Pred boxes: {total_preds}")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP50-95: {map50_95:.4f}")
    print(f"Precision@conf={args.conf}: {ap50_prec:.4f}")
    print(f"Recall@conf={args.conf}: {ap50_rec:.4f}")

    if args.out:
        out = {
            "gt_boxes": total_gt,
            "pred_boxes": total_preds,
            "map50": map50,
            "map50_95": map50_95,
            "conf_threshold": args.conf,
            "precision_at_conf": ap50_prec,
            "recall_at_conf": ap50_rec,
            "ap_by_iou": aps,
        }
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
