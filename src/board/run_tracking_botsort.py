#!/usr/bin/env python3
"""Ultralytics built-in Bot-SORT tracker + MOTA evaluation for VISEM."""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

import mmsc_modules
from ultralytics import YOLO


mmsc_modules.register()
if not hasattr(sys.modules.get("__main__", None), "NWDBboxLoss"):
    sys.modules["__main__"].NWDBboxLoss = mmsc_modules.NWDBboxLoss


MODEL = os.environ.get(
    "MODEL",
    "/root/code/runs/detect/runs/visem_tracking/talp_r45n10/weights/epoch24.pt",
)
SRC = Path("/root/code/VISEM_Tracking_Train_v4/Train")
MOT_GT = Path("/root/code/datasets/visem_tracking/mot_gt")
VAL_VIDS = os.environ.get("VAL_VIDS", "29,36,54,82").split(",")
OUT_DIR = Path("/root/code/runs/visem_tracking/botsort_track_results")

CONF = float(os.environ.get("CONF", "0.20"))
CONF_ALPHA = float(os.environ.get("CONF_ALPHA", "1.0"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
CLASSES = os.environ.get("CLASSES", None)
TRACKER_YAML = os.environ.get("TRACKER_YAML", "botsort.yaml")
NMS_IOU = float(os.environ.get("NMS_IOU", "0.7"))
MAX_DET = int(os.environ.get("MAX_DET", "2000"))


def parse_mot(fp):
    dets = defaultdict(list)
    for line in Path(fp).read_text().strip().splitlines():
        if not line.strip():
            continue
        p = line.strip().replace(",", " ").split()
        if len(p) < 6:
            continue
        dets[int(float(p[0]))].append({
            "id": int(float(p[1])),
            "bbox": [float(p[2]), float(p[3]), float(p[4]), float(p[5])],
        })
    return dets


def ciou(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def mot_metrics(gt, pred, iou_thr=0.5):
    frames = sorted(set(gt) | set(pred))
    tgt = ttp = tfp = tfn = tidsw = msum = mcnt = 0
    idmap = {}
    gtf = defaultdict(int)
    gtm = defaultdict(int)
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        tgt += len(g)
        for x in g:
            gtf[x["id"]] += 1
        if not g or not p:
            tfp += len(p)
            tfn += len(g)
            continue
        m = np.zeros((len(g), len(p)))
        for i, gi in enumerate(g):
            for j, pj in enumerate(p):
                m[i, j] = ciou(gi["bbox"], pj["bbox"])
        mg, mp = set(), set()
        pairs = sorted(
            [(m[i, j], i, j) for i in range(len(g)) for j in range(len(p)) if m[i, j] >= iou_thr],
            reverse=True,
        )
        for v, i, j in pairs:
            if i not in mg and j not in mp:
                mg.add(i)
                mp.add(j)
                msum += v
                mcnt += 1
                gi, pj = g[i]["id"], p[j]["id"]
                gtm[gi] += 1
                prev = idmap.get(gi)
                if prev is not None and prev != pj:
                    tidsw += 1
                idmap[gi] = pj
        ttp += len(mg)
        tfp += len(p) - len(mp)
        tfn += len(g) - len(mg)
    mota = 1 - (tfp + tfn + tidsw) / max(tgt, 1)
    motp = msum / max(mcnt, 1)
    idtp = ttp - tidsw
    idf1 = 2 * idtp / max(2 * idtp + tfp + tfn + 2 * tidsw, 1)
    mt = sum(1 for t, tt in gtf.items() if gtm.get(t, 0) >= 0.8 * tt) / max(len(gtf), 1)
    ml = sum(1 for t, tt in gtf.items() if gtm.get(t, 0) <= 0.2 * tt) / max(len(gtf), 1)
    return {"MOTA": mota, "MOTP": motp, "IDF1": idf1, "MT": mt, "ML": ml,
            "FP": tfp, "FN": tfn, "IDSW": tidsw, "GT": tgt, "TP": ttp}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading: {MODEL}")
    model = YOLO(MODEL)
    model.model.fuse = lambda verbose=False: model.model
    model.model.names = {0: "sperm"}

    for vid in VAL_VIDS:
        vp = SRC / vid / f"{vid}.mp4"
        if not vp.exists():
            print(f"SKIP {vid}")
            continue
        ld = OUT_DIR / vid / "labels"
        ld.mkdir(parents=True, exist_ok=True)
        for f in ld.iterdir():
            f.unlink()

        print(f"\nTracking {vid} with Bot-SORT...")
        results = model.track(
            source=str(vp),
            conf=CONF ** CONF_ALPHA,
            iou=NMS_IOU,
            classes=None if CLASSES is None else [int(x) for x in CLASSES.split(",")],
            augment=(os.environ.get("AUGMENT", "0") == "1"),
            max_det=MAX_DET,
            stream=True,
            device=0,
            imgsz=IMGSZ,
            tracker=TRACKER_YAML,
            persist=True,
            verbose=False,
        )
        for fi, r in enumerate(results):
            ih, iw = r.orig_shape
            if r.boxes is None or r.boxes.id is None:
                continue
            xywh = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            with open(ld / f"{vid}_{fi + 1}.txt", "w") as f:
                for k in range(len(xywh)):
                    cx = float(xywh[k, 0] / iw)
                    cy = float(xywh[k, 1] / ih)
                    w = float(xywh[k, 2] / iw)
                    h = float(xywh[k, 3] / ih)
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {int(ids[k])} 1.0\n")

    print("\n" + "=" * 60)
    am = []
    for vid in VAL_VIDS:
        gp = MOT_GT / f"{vid}.txt"
        ld = OUT_DIR / vid / "labels"
        if not gp.exists() or not ld.exists():
            continue
        gt = parse_mot(str(gp))
        pred = defaultdict(list)
        for lf in sorted(ld.iterdir()):
            try:
                frame = int(lf.stem.split("_")[-1])
            except ValueError:
                continue
            for line in lf.read_text().strip().splitlines():
                p = line.strip().split()
                if len(p) < 6:
                    continue
                xc, yc, bw, bh = map(float, p[1:5])
                tid = int(float(p[5]))
                pred[frame].append({"id": tid, "bbox": [xc, yc, bw, bh]})
        m = mot_metrics(gt, pred)
        print(f"  Video {vid}: MOTA={m['MOTA']:.3f} MOTP={m['MOTP']:.3f} IDF1={m['IDF1']:.3f} "
              f"FP={m['FP']} FN={m['FN']} IDSW={m['IDSW']}")
        am.append(m["MOTA"])
    print(f"\n  Mean MOTA: {sum(am) / len(am):.3f}")


if __name__ == "__main__":
    main()
