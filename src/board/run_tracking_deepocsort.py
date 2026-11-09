#!/usr/bin/env python3
"""DeepOCSORT-style tracker for VISEM.

Cascade association: high-confidence detections first with IoU+appearance, then
low-confidence detections with IoU only, then new-track initiation. Motion is a
simple constant-velocity predictor; appearance uses the trained sperm ReID.
"""

import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

import mmsc_modules
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy


mmsc_modules.register()

MODEL = os.environ.get(
    "MODEL",
    "/root/code/runs/detect/runs/visem_official/yolo11n_mmsc_fusion_v8_official/weights/best.pt",
)
REID_MODEL = os.environ.get("REID_MODEL", "/root/code/runs/visem_official/sperm_reid_v2.pt")
SRC = Path("/root/code/VISEM_Tracking_Train_v4/Train")
MOT_GT = Path("/root/code/datasets/visem_tracking/mot_gt")
VAL_VIDS = os.environ.get("VAL_VIDS", "82,60,54,52").split(",")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/root/code/runs/visem_tracking/deepocsort_results"))
PER_VID_CONF = {}
for _item in os.environ.get("PER_VID_CONF", "").split(","):
    _item = _item.strip()
    if ":" in _item:
        _vid, _conf = _item.split(":", 1)
        PER_VID_CONF[_vid.strip()] = float(_conf)
PER_VID_NMS = {}
for _item in os.environ.get("PER_VID_NMS", "").split(","):
    _item = _item.strip()
    if ":" in _item:
        _vid, _iou = _item.split(":", 1)
        PER_VID_NMS[_vid.strip()] = float(_iou)

CONF = float(os.environ.get("CONF", "0.30"))
NMS_IOU = float(os.environ.get("NMS_IOU", "0.5"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
MAX_DET = int(os.environ.get("MAX_DET", "500"))
AUGMENT = int(os.environ.get("AUGMENT", "0"))
SOFT_NMS = int(os.environ.get("SOFT_NMS", "0"))
SOFT_SIGMA = float(os.environ.get("SOFT_SIGMA", "0.5"))
SOFT_IOU = float(os.environ.get("SOFT_IOU", "0.5"))
RAW_CONF = float(os.environ.get("RAW_CONF", "0.05"))
RAW_IOU = float(os.environ.get("RAW_IOU", "0.99"))

MOTION_GATE = int(os.environ.get("MOTION_GATE", "0"))
MOTION_CONF = float(os.environ.get("MOTION_CONF", "0.10"))
MOTION_THRESH = int(os.environ.get("MOTION_THRESH", "15"))
MOTION_DILATE = int(os.environ.get("MOTION_DILATE", "5"))
MOTION_MIN_RATIO = float(os.environ.get("MOTION_MIN_RATIO", "0.05"))
MOTION_HIGH_CONF = float(os.environ.get("MOTION_HIGH_CONF", "0.50"))

TRACK_HIGH = float(os.environ.get("TRACK_HIGH", "0.5"))
TRACK_LOW = float(os.environ.get("TRACK_LOW", "0.2"))
NEW_TRACK = float(os.environ.get("NEW_TRACK", "0.5"))
MIN_HITS = int(os.environ.get("MIN_HITS", "2"))
MAX_AGE = int(os.environ.get("MAX_AGE", "60"))
MATCH_IOU = float(os.environ.get("MATCH_IOU", "0.3"))
LOW_MATCH_IOU = float(os.environ.get("LOW_MATCH_IOU", "0.2"))
LAMBDA_APP = float(os.environ.get("LAMBDA_APP", "0.1"))
APP_GATE = float(os.environ.get("APP_GATE", "0.0"))
APP_CONSIST = float(os.environ.get("APP_CONSIST", "0.0"))
VERY_LOW_CONF = float(os.environ.get("VERY_LOW_CONF", "-1"))
VERY_LOW_DIST = float(os.environ.get("VERY_LOW_DIST", "5.0"))
VERY_LOW_IOU = float(os.environ.get("VERY_LOW_IOU", "0.5"))

# Optional box-size calibration: the student detector systematically predicts
# boxes that are too small on the large-object videos, which puts the matched
# IoU just under the 0.5 threshold.  Scaling w/h about the box centre (as
# `analyze_detector_workpoint.py --box-scales` measures) is a pure post-process
# step with no extra inference cost.  Calibrate on the dev videos, report on the
# holdout.
BOX_SCALE = float(os.environ.get("BOX_SCALE", "1.0"))


# Size-conditioned version of the same correction: a per-box factor looked up
# from the box side instead of one constant.  Measured on
# ablation_dyhead30_448 at 448, the matched det/GT side ratio is monotone in
# object size inside every video (82: 1.430 / 1.345 / 1.113 / 0.959 across the
# 0-8 / 8-12 / 12-16 / 16-20 px bands; 54: 0.997 / 0.786 / 0.712 / 0.645 / 0.545
# / 0.363), i.e. the box regression is *compressed* and emits roughly one size
# whatever the object is.  A single scalar can only pivot that curve about one
# size, so it has to over-correct the small objects to fix the large ones.
#
# Knots are ``size_px:factor`` in MODEL-INPUT pixels, piecewise-linear and flat
# outside the ends, and they multiply on top of BOX_SCALE.  Fit them with
# sweep_box_curve.py, which scores the curve against the scalar offline on the
# same 5%-of-frames calibration protocol.  Empty (the default) leaves the scalar
# path bit-identical to what every earlier result used.
def _parse_box_curve(spec: str):
    xs, ys = [], []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        size, factor = item.split(":")
        xs.append(float(size))
        ys.append(float(factor))
    if not xs:
        return None
    order = np.argsort(xs)
    return np.asarray(xs, dtype=np.float64)[order], np.asarray(ys, dtype=np.float64)[order]


BOX_SCALE_CURVE = _parse_box_curve(os.environ.get("BOX_SCALE_CURVE", ""))
# Run the detector on every DET_STRIDE-th frame and feed the tracker an empty
# detection list in between, so its Kalman prediction carries the tracks across
# the skipped frames.  Sperm move about 1 px per frame and the videos are 50 FPS,
# so the detector does not need every frame: measured on this pipeline, striding
# by 3 or 4 leaves dev MOTA unchanged (0.612 -> 0.618/0.620) while cutting the
# detector cost by that factor.  See sweep_detection_stride.py.
#
# DET_STRIDE == 1 keeps the original ultralytics stream path untouched, so no
# existing result can be disturbed by this option.  Set DET_LOOP=1 to force the
# frame-loop path at stride 1 as well, which is how you verify the rewrite
# against the old numbers.
DET_STRIDE = max(1, int(os.environ.get("DET_STRIDE", "1")))
DET_LOOP = int(os.environ.get("DET_LOOP", "0")) or DET_STRIDE > 1

# Skip the OSNet appearance embedding entirely.  This is not the same as
# LAMBDA_APP=0: that only down-weights the appearance term in the assignment
# cost while still running the network on every detection.  With NO_REID=1 the
# forward pass is skipped (returning None, which Track/Tracker.match already
# handle) and the association becomes pure IoU on the Kalman prediction.
#
# Worth measuring because MOTA here is dominated by FP/FN, not IDSW: IDSW is
# ~0.1-0.2% of GT, so dropping appearance can only cost a few thousandths while
# removing the most expensive CPU stage from the board pipeline.
NO_REID = int(os.environ.get("NO_REID", "0"))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SpermReIDModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)


def load_reid(path, device):
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("reid_kind") == "osnet":
        from osnet_model import osnet_ibn_x1_0

        model = osnet_ibn_x1_0(
            num_classes=ckpt.get("num_ids", 1),
            pretrained=False,
            loss="softmax",
            feature_dim=ckpt.get("feature_dim", ckpt.get("embed_dim", 512)),
        )
        state = {k: v for k, v in ckpt["model"].items() if not k.startswith("classifier.")}
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return model

    model = SpermReIDModel(ckpt.get("embed_dim", 128))
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("classifier.")}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def reid_embedding(model, img_bgr, xyxy, device):
    if NO_REID:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
    x = (crop.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)[0].cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-6)


def xyxy2xywh_np(boxes):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def soft_nms(boxes, scores, iou_thr=0.5, sigma=0.5, conf_thr=0.0):
    """Gaussian soft-NMS, keeping boxes whose decayed score stays above conf_thr."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1].copy()
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        scores[order[1:]] *= np.exp(-(iou * iou) / sigma)
        order = order[1:]
        order = order[np.where(scores[order] >= conf_thr)[0]]
        if order.size > 0:
            order = order[np.argsort(scores[order])[::-1]]
    keep = np.asarray(keep, dtype=np.int64)
    return keep, scores[keep] if keep.size else scores[:0]


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def motion_overlap(mask, xyxy):
    """Fraction of a box area that overlaps a binarized motion mask."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    return float((roi > 0).mean())


def iou_matrix(A, B):
    """IoU between every (T, 4) and (D, 4) xywh box.

    The association cost used to be built with a Python double loop.  Track
    pruning only removes tracks past MAX_AGE, so the live set grows to
    roughly (new tracks per detection frame) x (MAX_AGE / stride) -- hundreds of
    tracks in practice.  At that size the double loop costs ~550 ms per frame on
    the RK3568 A55 and dominates the whole pipeline; this version does the same
    arithmetic in a broadcast expression.
    """
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), np.float32)
    ax1, ay1 = A[:, 0] - A[:, 2] / 2, A[:, 1] - A[:, 3] / 2
    ax2, ay2 = A[:, 0] + A[:, 2] / 2, A[:, 1] + A[:, 3] / 2
    bx1, by1 = B[:, 0] - B[:, 2] / 2, B[:, 1] - B[:, 3] / 2
    bx2, by2 = B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2
    iw = np.maximum(0.0, np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]))
    ih = np.maximum(0.0, np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]))
    inter = iw * ih
    union = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


class Track:
    def __init__(self, tid, xywh, feat, conf):
        self.id = tid
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.vel = np.zeros(4, dtype=np.float32)
        self.features = [feat] if feat is not None else []
        self.conf = conf
        self.hits = 1
        self.age = 0

    def predict(self):
        self.xywh = self.xywh + self.vel

    def update(self, xywh, feat, conf):
        new = np.asarray(xywh, dtype=np.float32)
        self.vel = 0.8 * self.vel + 0.2 * (new - self.xywh)
        self.xywh = new
        self.conf = conf
        if feat is not None:
            self.features.append(feat)
            if len(self.features) > 30:
                self.features.pop(0)
        self.hits += 1
        self.age = 0

    def appearance(self):
        if not self.features:
            return None
        return np.mean(np.stack(self.features, axis=0), axis=0)

    def appearance_consistency(self):
        """Mean pairwise cosine similarity of the feature gallery."""
        if len(self.features) < 2:
            return 1.0
        f = np.stack(self.features, axis=0)
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-6)
        sim = f @ f.T
        triu = np.triu_indices(len(self.features), 1)
        if triu[0].size == 0:
            return 1.0
        return float(sim[triu].mean())


class Tracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        if self.tracks:
            # Batch the constant-velocity step.  `for t: t.xywh = t.xywh + t.vel`
            # is one numpy call per track -- at ~100 live tracks that is ~200
            # numpy calls per frame with no vector work to amortise, ~5 ms on the
            # RK3568 A55.  Two np.stack calls plus one add replace all of it.
            cur = np.stack([t.xywh for t in self.tracks])
            cur += np.stack([t.vel for t in self.tracks])
            for t, v in zip(self.tracks, cur):
                t.xywh = v.copy()      # own the buffer; avoids aliasing `cur`
                t.age += 1

        high = [d for d in detections if d["conf"] >= TRACK_HIGH]
        low = [d for d in detections if TRACK_LOW <= d["conf"] < TRACK_HIGH]

        matched, unr, unc = self.match(self.tracks, high, use_app=True, thresh=MATCH_IOU)
        for r, c in matched:
            self.tracks[r].update(high[c]["xywh"], high[c]["feat"], high[c]["conf"])

        rem_tracks = [self.tracks[i] for i in unr]
        rem_high = [high[j] for j in unc]
        matched2, unr2, unc2 = self.match(rem_tracks, low, use_app=False, thresh=LOW_MATCH_IOU)
        for r, c in matched2:
            rem_tracks[r].update(low[c]["xywh"], low[c]["feat"], low[c]["conf"])

        if VERY_LOW_CONF >= 0:
            very_low = [d for d in detections if VERY_LOW_CONF <= d["conf"] < TRACK_LOW]
            rem_tracks = [rem_tracks[i] for i in unr2]
            matched3, _, _ = self.match_very_low(rem_tracks, very_low)
            for r, c in matched3:
                rem_tracks[r].update(very_low[c]["xywh"], very_low[c]["feat"], very_low[c]["conf"])

        for j in unc:
            if high[j]["conf"] >= NEW_TRACK:
                self.tracks.append(Track(self.next_id, high[j]["xywh"], high[j]["feat"], high[j]["conf"]))
                self.next_id += 1

        live = []
        for t in self.tracks:
            if t.age > MAX_AGE:
                continue
            if t.hits >= MIN_HITS and t.age == 0:
                if APP_CONSIST > 0 and t.appearance_consistency() < APP_CONSIST:
                    continue
                live.append(t)
        self.tracks = [t for t in self.tracks if t.age <= MAX_AGE]
        return live

    def match_very_low(self, tracks, dets):
        """Match very-low-confidence detections to existing tracks by proximity."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost = np.full((len(tracks), len(dets)), 1e6, dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                dx = abs(t.xywh[0] - d["xywh"][0])
                dy = abs(t.xywh[1] - d["xywh"][1])
                if dx > VERY_LOW_DIST or dy > VERY_LOW_DIST:
                    continue
                iou = iou_xywh(t.xywh, d["xywh"])
                if iou < VERY_LOW_IOU:
                    continue
                cost[i, j] = dx + dy
        rows, cols = linear_sum_assignment(cost)
        matched = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < 1e5]
        matched_r = {r for r, _ in matched}
        matched_c = {c for _, c in matched}
        unr = [i for i in range(len(tracks)) if i not in matched_r]
        unc = [j for j in range(len(dets)) if j not in matched_c]
        return matched, unr, unc

    def match(self, tracks, dets, use_app, thresh):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        M = iou_matrix(np.stack([t.xywh for t in tracks]),
                       np.stack([d["xywh"] for d in dets]))
        valid = M >= thresh
        # Same arithmetic as before: unmatched pairs cost 1e6, and a pair with no
        # usable appearance keeps the neutral app_cost of 0.5.
        cost = np.full(M.shape, 1e6, dtype=np.float32)
        cost[valid] = LAMBDA_APP * 0.5 + (1.0 - LAMBDA_APP) * (1.0 - M[valid])
        if use_app and LAMBDA_APP > 0.0:
            for i, t in enumerate(tracks):
                fa = t.appearance()
                if fa is None:
                    continue
                for j, d in enumerate(dets):
                    if not valid[i, j] or d["feat"] is None:
                        continue
                    cos = float(np.dot(fa, d["feat"]))
                    if cos < APP_GATE:
                        cost[i, j] = 1e6
                        continue
                    cost[i, j] = LAMBDA_APP * (1.0 - cos) + (1.0 - LAMBDA_APP) * (1.0 - M[i, j])

        rows, cols = linear_sum_assignment(cost)
        matched = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < 1e5]
        matched_r = {r for r, _ in matched}
        matched_c = {c for _, c in matched}
        unr = [i for i in range(len(tracks)) if i not in matched_r]
        unc = [j for j in range(len(dets)) if j not in matched_c]
        return matched, unr, unc


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
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        tgt += len(g)
        if not g or not p:
            tfp += len(p)
            tfn += len(g)
            continue
        m = np.zeros((len(g), len(p)))
        for i, gi in enumerate(g):
            for j, pj in enumerate(p):
                m[i, j] = ciou(gi["bbox"], pj["bbox"])
        mg, mp = set(), set()
        pairs = sorted([(m[i, j], i, j) for i in range(len(g)) for j in range(len(p)) if m[i, j] >= iou_thr], reverse=True)
        for v, i, j in pairs:
            if i not in mg and j not in mp:
                mg.add(i)
                mp.add(j)
                msum += v
                mcnt += 1
                gid, pid = g[i]["id"], p[j]["id"]
                prev = idmap.get(gid)
                if prev is not None and prev != pid:
                    tidsw += 1
                idmap[gid] = pid
        ttp += len(mg)
        tfp += len(p) - len(mp)
        tfn += len(g) - len(mg)
    mota = 1 - (tfp + tfn + tidsw) / max(tgt, 1)
    motp = msum / max(mcnt, 1)
    return {"MOTA": mota, "MOTP": motp, "FP": tfp, "FN": tfn, "IDSW": tidsw, "GT": tgt, "TP": ttp}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    model = YOLO(MODEL)
    model.model.fuse = lambda verbose=False: model.model
    model.model.names = {0: "sperm"}
    reid = load_reid(REID_MODEL, device)
    predict_conf = MOTION_CONF if MOTION_GATE else (RAW_CONF if SOFT_NMS else CONF)
    predict_max_det = max(MAX_DET, 5000) if SOFT_NMS else MAX_DET

    for vid in VAL_VIDS:
        vid_conf = PER_VID_CONF.get(vid, CONF)
        vid_nms = PER_VID_NMS.get(vid, NMS_IOU)
        predict_conf = MOTION_CONF if MOTION_GATE else (RAW_CONF if SOFT_NMS else vid_conf)
        predict_iou = RAW_IOU if SOFT_NMS else vid_nms
        vp = SRC / vid / f"{vid}.mp4"
        if not vp.exists():
            continue
        ld = OUT_DIR / vid / "labels"
        ld.mkdir(parents=True, exist_ok=True)
        for f in ld.iterdir():
            f.unlink()

        tracker = Tracker()
        prev_gray = None
        n_det = 0

        def process(fi, frame_bgr, boxes):
            """Track one frame and write its labels.  boxes=None means 'no detection this frame'."""
            nonlocal prev_gray
            ih, iw = frame_bgr.shape[:2]
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            motion_mask = None
            if MOTION_GATE and prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diff = cv2.GaussianBlur(diff, (5, 5), 0)
                _, motion_mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
                if MOTION_DILATE > 0:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (MOTION_DILATE * 2 + 1, MOTION_DILATE * 2 + 1)
                    )
                    motion_mask = cv2.dilate(motion_mask, kernel)

            detections = []
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                if SOFT_NMS:
                    keep, conf = soft_nms(xyxy, conf, SOFT_IOU, SOFT_SIGMA, vid_conf)
                    xyxy = xyxy[keep]
                    xywh_np = xyxy2xywh_np(xyxy)
                else:
                    xywh_np = xyxy2xywh_np(xyxy)
                for k in range(len(xywh_np)):
                    if MOTION_GATE and motion_mask is not None:
                        if float(conf[k]) < MOTION_HIGH_CONF and motion_overlap(motion_mask, xyxy[k]) < MOTION_MIN_RATIO:
                            continue
                    feat = reid_embedding(reid, frame_bgr, xyxy[k], device)
                    detections.append({"xywh": xywh_np[k], "feat": feat, "conf": float(conf[k])})
            tracks = tracker.update(detections)
            with open(ld / f"{vid}_{fi + 1}.txt", "w") as f:
                for t in tracks:
                    cx = t.xywh[0] / iw
                    cy = t.xywh[1] / ih
                    scale = BOX_SCALE
                    if BOX_SCALE_CURVE is not None:
                        letter = IMGSZ / max(ih, iw)
                        side = float(
                            np.sqrt(max(t.xywh[2] * iw * letter * t.xywh[3] * ih * letter, 1e-9))
                        )
                        scale *= float(np.interp(side, BOX_SCALE_CURVE[0], BOX_SCALE_CURVE[1]))
                    w = t.xywh[2] / iw * scale
                    h = t.xywh[3] / ih * scale
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {t.id} 1.0\n")
            prev_gray = gray

        if not DET_LOOP:
            # original path: ultralytics reads the video and predicts every frame
            results = model.predict(source=str(vp), conf=predict_conf, iou=predict_iou,
                                    max_det=predict_max_det, stream=True, device=0, imgsz=IMGSZ,
                                    augment=bool(AUGMENT), verbose=False)
            for fi, r in enumerate(results):
                process(fi, r.orig_img, r.boxes)
                n_det += 1
        else:
            # strided path: read frames ourselves so the detector really is skipped
            cap = cv2.VideoCapture(str(vp))
            fi = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if fi % DET_STRIDE == 0:
                    r = model.predict(source=frame, conf=predict_conf, iou=predict_iou,
                                      max_det=predict_max_det, device=0, imgsz=IMGSZ,
                                      augment=bool(AUGMENT), verbose=False)[0]
                    process(fi, frame, r.boxes)
                    n_det += 1
                else:
                    process(fi, frame, None)
                fi += 1
            cap.release()
            print(f"  {vid}: {fi} frames, detector ran on {n_det} "
                  f"(stride {DET_STRIDE} -> detection rate is 1/{DET_STRIDE} of the video rate)")

    print("\n" + "=" * 60)
    am = []
    tot = {"FP": 0, "FN": 0, "IDSW": 0, "GT": 0, "TP": 0}
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
            for line in lf.read_text().splitlines():
                p = line.split()
                if len(p) < 6:
                    continue
                pred[frame].append({"id": int(float(p[5])), "bbox": [float(p[1]), float(p[2]), float(p[3]), float(p[4])]})
        m = mot_metrics(gt, pred)
        for k in tot:
            tot[k] += m[k]
        print(f"  Video {vid}: MOTA={m['MOTA']:.3f} MOTP={m['MOTP']:.3f} "
              f"FP={m['FP']} FN={m['FN']} IDSW={m['IDSW']}")
        am.append(m["MOTA"])
    print(f"\n  Mean MOTA: {sum(am) / max(len(am), 1):.3f}")
    print(f"  Total FP={tot['FP']} FN={tot['FN']} IDSW={tot['IDSW']} GT={tot['GT']}")


if __name__ == "__main__":
    main()
