#!/usr/bin/env python3
"""Board-side single-model head-raw inference: RKNN backbone+head, CPU DFL/decode.

Why a new script
----------------
The two existing board scripts do not match this export:
  * rknn_headless_detect_raw.py runs TWO models (headless INT8 + head-raw FP16);
  * rknn_lite_bench_head_raw.py takes p3/p4/p5 feature maps as input, not images.

The export_onnx_deploy.py --head-raw model is a single graph: image in, one
(1, 4*reg_max+nc, H, W) tensor per detection scale out.  This script feeds it an
image, decodes on the CPU exactly the way Ultralytics' Detect._inference does
(DFL softmax expectation -> dist2bbox -> stride scaling -> sigmoid), and reports
how many boxes survive.

It exists because the INT8 all-zero failure could only be diagnosed by looking at
output VALUES, never at the conversion log.  Run the FP16 and INT8 .rknn through
this and compare the detection counts.

    python3 rknn_headraw_detect.py --self-test

    python3 rknn_headraw_detect.py --rknn dyhead30_448_headraw_int8.rknn \
        --imgsz 448 --img_dir val50 --conf 0.25

    python3 rknn_headraw_detect.py --rknn dyhead30_448_headraw_int8.rknn \
        --imgsz 448 --image val50/13_frame_1018.jpg --runs 100
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Local cross-check convenience: prefer a bundled ultralytics source tree.  The
# board does not have ultralytics, where self_test() reports that it cannot run.
for _extra in (Path(__file__).resolve().parent / "ultralytics-8.3.163", Path(__file__).resolve().parent):
    if _extra.is_dir() and str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


def letterbox(img, size):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dy, dx = (size - nh) // 2, (size - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img
    return canvas, r, dx, dy


_ANCHOR_CACHE = {}


def make_anchors(shapes, strides, cell=0.5):
    """Row-major (y, x) anchor grid per scale, matching Ultralytics make_anchors.

    Returns anchors as (2, N) and per-anchor strides as (1, N), which is the
    layout Detect._inference keeps after its `transpose(0, 1)`.
    """
    key = (tuple(map(tuple, shapes)), tuple(strides), cell)
    if key in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[key]
    pts, strd = [], []
    for (h, w), s in zip(shapes, strides):
        sx = np.arange(w, dtype=np.float32) + cell
        sy = np.arange(h, dtype=np.float32) + cell
        gx, gy = np.meshgrid(sx, sy)  # 'xy' default: gx[i, j] = sx[j]
        pts.append(np.stack([gx, gy], -1).reshape(-1, 2))
        strd.append(np.full((h * w, 1), s, dtype=np.float32))
    out = (np.concatenate(pts, 0).T, np.concatenate(strd, 0).T)
    _ANCHOR_CACHE[key] = out          # the grid is fixed for a fixed input size
    return out


def _split_layout(a, reg_max=16):
    """One head-raw output -> ((C, H, W) float32, (H, W))."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4 and a.shape[1] == 4 * reg_max + 1:
        a = a[0]
    elif a.ndim == 4 and a.shape[-1] == 4 * reg_max + 1:
        a = np.transpose(a[0], (2, 0, 1))
    if a.ndim != 3 or a.shape[0] != 4 * reg_max + 1:
        raise ValueError(f"unexpected head-raw shape {np.asarray(a).shape}")
    return a, a.shape[1:]


def class_scores_fast(outs, reg_max=16):
    """Sigmoid class scores (N,) without materialising the box channels.

    The class channel is 1 of 4*reg_max+1; the original code concatenated the
    whole (65, N) logit block just to read it.
    """
    parts, shapes = [], []
    for o in outs:
        a, (h, w) = _split_layout(o, reg_max)
        shapes.append((h, w))
        # keep the class axis: reshape (nc, H, W) -> (nc, H*W), not flat, or the
        # max over classes would collapse the whole tensor to a scalar
        parts.append(a[4 * reg_max:].reshape(a.shape[0] - 4 * reg_max, -1))
    cls = np.concatenate(parts, axis=1)
    return (1.0 / (1.0 + np.exp(-np.clip(cls, -60.0, 60.0)))).max(axis=0), shapes


def boxes_for(outs, imgsz, keep, reg_max=16):
    """cxcywh (4, K) for the selected global anchor indices.

    Decodes per scale, taking only the surviving anchors, so the full
    concatenation and the DFL over all 4116 anchors never happen.
    """
    norm, counts = [], []
    for o in outs:
        a, (h, w) = _split_layout(o, reg_max)
        norm.append(a)
        counts.append(h * w)
    shapes = [a.shape[1:] for a in norm]
    bounds = np.concatenate([[0], np.cumsum(counts)])
    anchors, stride_t = make_anchors(shapes, [float(imgsz) / s[0] for s in shapes])

    out = np.empty((4, keep.size), np.float32)
    arange = np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)
    for s, a in enumerate(norm):
        lo, hi = bounds[s], bounds[s + 1]
        m = (keep >= lo) & (keep < hi)
        if not m.any():
            continue
        loc = keep[m] - lo
        box = a[: 4 * reg_max].reshape(4, reg_max, -1)[:, :, loc]
        box = box - box.max(axis=1, keepdims=True)
        p = np.exp(box)
        p /= p.sum(axis=1, keepdims=True)
        dist = (p * arange).sum(axis=1)
        anc, st = anchors[:, keep[m]], stride_t[:, keep[m]]
        x1y1 = anc - dist[:2]
        x2y2 = anc + dist[2:]
        out[:, m] = np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], 0) * st
    return out


def _concat_outputs(outs, reg_max=16):
    """head-raw tensors -> ((C, N) logits, [(H, W) per scale])."""
    shapes, parts = [], []
    for o in outs:
        a = np.asarray(o, dtype=np.float32)
        if a.ndim == 4 and a.shape[1] == 4 * reg_max + 1:
            a = a[0]
        elif a.ndim == 4 and a.shape[-1] == 4 * reg_max + 1:
            a = np.transpose(a[0], (2, 0, 1))
        if a.ndim != 3 or a.shape[0] != 4 * reg_max + 1:
            raise ValueError(f"unexpected head-raw shape {np.asarray(o).shape}")
        shapes.append(a.shape[1:])
        parts.append(a.reshape(a.shape[0], -1))
    return np.concatenate(parts, axis=1), shapes


def class_scores(outs, reg_max=16):
    """Sigmoid class scores only.  The class and box channels are independent, so
    the score can be computed for every anchor while the DFL (the expensive part)
    is deferred to the anchors that actually pass the threshold."""
    cat, shapes = _concat_outputs(outs, reg_max)
    cls = cat[4 * reg_max:]
    return (1.0 / (1.0 + np.exp(-np.clip(cls, -60.0, 60.0)))).max(axis=0), cat, shapes


def decode_from(cat, shapes, imgsz, reg_max=16, idx=None):
    """DFL + dist2bbox from an already-concatenated (C, N) logit array."""
    anchors, stride_t = make_anchors(shapes, [float(imgsz) / s[0] for s in shapes])
    if idx is not None:
        anchors, stride_t = anchors[:, idx], stride_t[:, idx]
    box, cls = cat[: 4 * reg_max], cat[4 * reg_max:]
    if idx is not None:
        box, cls = box[:, idx], cls[:, idx]

    b = box.reshape(4, reg_max, box.shape[1])
    b = b - b.max(axis=1, keepdims=True)
    p = np.exp(b)
    p /= p.sum(axis=1, keepdims=True)
    dist = (p * np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)).sum(axis=1)

    x1y1 = anchors - dist[:2]
    x2y2 = anchors + dist[2:]
    cxcywh = np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], 0) * stride_t
    scores = 1.0 / (1.0 + np.exp(-np.clip(cls, -60.0, 60.0)))
    return cxcywh, scores.max(axis=0)


def decode_raw(outs, imgsz, reg_max=16, idx=None):
    """head-raw tensors -> (cxcywh (4,N) in imgsz pixels, sigmoid scores (N,)).

    Mirrors Detect._inference:
        cat([xi.view(b, no, -1)]) -> split(box, cls)
        dfl(box) -> dist2bbox(..., xywh=True) -> * strides
        cls.sigmoid()

    `idx` restricts the DFL to a subset of anchors (the ones above threshold);
    the returned box columns then correspond to `idx`.
    """
    cat, shapes = _concat_outputs(outs, reg_max)
    return decode_from(cat, shapes, imgsz, reg_max, idx)


def iou_xyxy_matrix(B):
    """Full (N, N) IoU matrix for xyxy boxes, in one broadcast expression."""
    x1, y1, x2, y2 = B[:, 0][:, None], B[:, 1][:, None], B[:, 2][:, None], B[:, 3][:, None]
    X1, Y1, X2, Y2 = B[:, 0][None, :], B[:, 1][None, :], B[:, 2][None, :], B[:, 3][None, :]
    iw = np.maximum(0.0, np.minimum(x2, X2) - np.maximum(x1, X1))
    ih = np.maximum(0.0, np.minimum(y2, Y2) - np.maximum(y1, Y1))
    inter = iw * ih
    a = np.maximum(0.0, B[:, 2] - B[:, 0]) * np.maximum(0.0, B[:, 3] - B[:, 1])
    union = a[:, None] + a[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes, scores, iou_thr):
    """Greedy NMS.

    The straightforward version recomputed IoU with ~8 numpy calls per kept box.
    On the RK3568 A55 a numpy call costs tens of microseconds regardless of size,
    so ~40 boxes became 19 ms -- more than half the whole decode stage.  One
    broadcast matrix plus a plain-Python scan over its rows removes that
    per-call overhead entirely.
    """
    n = len(boxes)
    if n == 0:
        return []
    # One broadcast for the whole IoU matrix, then two numpy calls per kept box
    # (the mask and its in-place OR).  A pure-Python inner loop was tried first
    # and is worse: the candidate count reaches several hundred, so O(n^2)
    # interpreted iterations cost more than the vectorised mask.
    M = iou_xyxy_matrix(boxes)
    dead = np.zeros(n, dtype=bool)
    keep = []
    for i in np.argsort(-np.asarray(scores)).tolist():
        if dead[i]:
            continue
        keep.append(i)
        np.logical_or(dead, M[i] > iou_thr, out=dead)
    return keep


def postprocess(outs, imgsz, conf=0.25, iou=0.45):
    # Threshold first, decode second: the DFL softmax over all 4116 anchors is
    # what made this cost ~46 ms on the RK3568 A55, while only ~150 anchors ever
    # survive.  Class and box channels are independent, so this is exact.
    scores, shapes = class_scores_fast(outs)
    keep = np.flatnonzero(scores > conf)
    if keep.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), scores
    cxcywh = boxes_for(outs, imgsz, keep)
    cx, cy, w, h = cxcywh
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    sc = scores[keep]
    idx = nms(boxes, sc, iou)
    return boxes[idx], sc[idx], scores


def self_test():
    """Compare the numpy decode against Ultralytics' own DFL/dist2bbox."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("self-test skipped: torch/ultralytics not installed here "
              "(run this on the server or the toolkit VM, not on the board)")
        return 0
    import torch
    from ultralytics.nn.modules import DFL
    from ultralytics.utils.tal import dist2bbox
    from ultralytics.utils.tal import make_anchors as ul_make_anchors

    reg_max, imgsz = 16, 448
    shapes = [(56, 56), (28, 28), (14, 14)]
    strides = [8.0, 16.0, 32.0]
    rng = np.random.default_rng(0)
    outs = [rng.normal(0, 1.5, (1, 4 * reg_max + 1, h, w)).astype(np.float32)
            for h, w in shapes]

    feats = [torch.from_numpy(o) for o in outs]
    anchors, stride_t = ul_make_anchors(feats, torch.tensor(strides), 0.5)
    anchors, stride_t = anchors.transpose(0, 1), stride_t.transpose(0, 1)
    cat = torch.cat([f.view(1, 4 * reg_max + 1, -1) for f in feats], 2)[0]
    box, cls = cat[: 4 * reg_max], cat[4 * reg_max:]
    ref = dist2bbox(DFL(reg_max)(box.unsqueeze(0))[0], anchors, xywh=True, dim=0) * stride_t
    ref_wh = ref.numpy()
    ref_scores = torch.sigmoid(cls).max(0).values.numpy()

    got_wh, got_scores = decode_raw(outs, imgsz, reg_max)
    d_wh = float(np.abs(got_wh - ref_wh).max())
    d_sc = float(np.abs(got_scores - ref_scores).max())
    assert d_wh < 1e-3, f"decode mismatch: max |dwh| = {d_wh}"
    assert d_sc < 1e-5, f"score mismatch: max |dscore| = {d_sc}"

    # postprocess() decodes only the anchors above threshold; that subset path
    # must give exactly the same boxes as the full decode.
    idx = np.flatnonzero(ref_scores > 0.5)
    if idx.size:
        sub_wh, sub_sc = decode_raw(outs, imgsz, reg_max, idx=idx)
        # compare against our own full decode, not the torch reference (which
        # already differs by ~1e-4 in float32).  The subset path walks a strided
        # copy instead of a contiguous array, so exp() takes a different SIMD
        # path and the results differ by ~2e-4 px -- six orders of magnitude
        # below anything that matters for 19-29 px targets.
        assert np.abs(sub_wh - got_wh[:, idx]).max() < 1e-3, "subset decode disagrees"
        assert np.abs(sub_sc - got_scores[idx]).max() < 1e-6, "subset scores disagree"

    # The fast path (class channel extraction + per-scale subset decode) must
    # reproduce the reference decode exactly.
    fast_scores, _ = class_scores_fast(outs, reg_max)
    assert np.abs(fast_scores - got_scores).max() < 1e-6, "fast class scores disagree"
    fidx = np.flatnonzero(fast_scores > 0.5)
    if fidx.size:
        fast_wh = boxes_for(outs, imgsz, fidx, reg_max)
        assert np.abs(fast_wh - got_wh[:, fidx]).max() < 1e-3, "fast boxes disagree"

    # NMS: the rewritten version must keep the same set as a naive reference.
    rng2 = np.random.default_rng(7)
    bx = np.concatenate([rng2.uniform(0, 400, (40, 2)),
                         rng2.uniform(20, 60, (40, 2))], 1).astype(np.float32)
    sc = rng2.random(40).astype(np.float32)
    got_keep = sorted(nms(bx, sc, 0.45))
    order = np.argsort(-sc)
    dead, ref = np.zeros(len(bx), bool), []
    I = iou_xyxy_matrix(bx)
    for i in order:
        if dead[i]:
            continue
        ref.append(int(i))
        dead |= I[i] > 0.45
    assert got_keep == sorted(ref), "NMS disagrees with the reference sweep"

    print(f"self-test OK: numpy decode matches ultralytics (max |dwh|={d_wh:.2e}, "
          f"max |dscore|={d_sc:.2e}, N={got_wh.shape[1]}); fast path and NMS agree")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--rknn", default=None)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--image", default=None)
    ap.add_argument("--img_dir", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.rknn:
        ap.error("--rknn is required")

    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    print("load_rknn:", rknn.load_rknn(args.rknn))
    print("init_runtime:", rknn.init_runtime(core_mask=0))

    if args.image:
        images = [args.image]
    elif args.img_dir:
        images = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))[: args.limit]
    else:
        raise SystemExit("pass --image or --img_dir")
    if not images:
        raise SystemExit("no images found")

    totals = []
    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"{os.path.basename(path)}: cannot read")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas, r, dx, dy = letterbox(rgb, args.imgsz)

        t0 = time.perf_counter()
        outs = None
        for _ in range(args.runs):
            outs = rknn.inference(inputs=[canvas[None, ...]], data_format="nhwc")
        dt = (time.perf_counter() - t0) / max(1, args.runs)

        boxes, sc, all_scores = postprocess(outs, args.imgsz, args.conf, args.iou)
        boxes = (boxes - np.array([dx, dy, dx, dy], np.float32)) / r
        totals.append(len(boxes))

        print(f"\n{os.path.basename(path)}   infer {dt * 1000:.1f} ms")
        print(f"  raw outputs {[tuple(np.asarray(o).shape) for o in outs]}")
        print(f"  cls sigmoid max {float(all_scores.max()):.4f}   "
              f">0.1 {int((all_scores > 0.1).sum())}   >0.25 {int((all_scores > 0.25).sum())}")
        print(f"  after NMS@{args.conf}: {len(boxes)} boxes")
        for i in sc.argsort()[::-1][:3]:
            b = boxes[i]
            print(f"    {sc[i]:.3f}  xyxy {b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}"
                  f"  wh {b[2] - b[0]:.1f}x{b[3] - b[1]:.1f}")

    if totals:
        print(f"\nimages {len(totals)}   boxes/image "
              f"min {min(totals)}  max {max(totals)}  mean {sum(totals) / len(totals):.1f}")
    rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
