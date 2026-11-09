#!/usr/bin/env python3
"""Array-state tracker with the exact semantics of rknn_headraw_pipeline.Tracker.

Motivation (measured on the RK3568 board)
-----------------------------------------
The reference pipeline builds one Python dict per detection, and each dict holds
a freshly allocated `np.array(..., np.float32)` box; the tracker then re-stacks
those arrays every frame.  With ~35 detections per frame that is ~70 numpy
allocations *per detected frame* before the tracker does any work -- and none of
it appears in the pipeline's reported stage timings, because the conversion loop
sits between the timed `nms` block and the timed `tracker.update` block.  On the
A55, where a tiny numpy call costs ~10-200 us, that untimed loop is the same
order as the whole NPU forward pass.

This module keeps the tracker's *math* identical (velocity prediction, high-conf
pass, low-conf pass on the leftovers, new tracks from unmatched high-conf
detections, `age <= max_age`, same Hungarian/greedy assignment) and only changes
the representation: boxes arrive as one (D, 4) float32 array plus a (D,) score
array, so the per-detection allocation disappears.

`verify_fasttrack.py` replays the board dumps of all four sessions through both
trackers and compares the emitted MOT labels element by element.
"""

from __future__ import annotations

import numpy as np

try:                                   # the board may not have scipy
    from scipy.optimize import linear_sum_assignment
except Exception:                      # pragma: no cover
    linear_sum_assignment = None


def iou_matrix(A, B):
    """IoU between every (T,4) and (D,4) xywh box -- reference formula."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), np.float32)
    ax1, ay1 = A[:, 0] - A[:, 2] / 2, A[:, 1] - A[:, 3] / 2
    ax2, ay2 = A[:, 0] + A[:, 2] / 2, A[:, 1] + A[:, 3] / 2
    bx1, by1 = B[:, 0] - B[:, 2] / 2, B[:, 1] - B[:, 3] / 2
    bx2, by2 = B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2
    iw = np.maximum(0.0, np.minimum(ax2[:, None], bx2[None, :])
                    - np.maximum(ax1[:, None], bx1[None, :]))
    ih = np.maximum(0.0, np.minimum(ay2[:, None], by2[None, :])
                    - np.maximum(ay1[:, None], by1[None, :]))
    inter = iw * ih
    union = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _assign(M, thresh):
    """Matched (row, col) pairs for IoU >= thresh -- reference decision rule."""
    valid = M >= thresh
    if not valid.any():
        return []
    if linear_sum_assignment is not None:
        cost = np.where(valid, 1.0 - M, 1e6)
        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]
    ii, jj = np.nonzero(valid)
    order = np.argsort(-M[ii, jj])
    ii, jj = ii[order].tolist(), jj[order].tolist()
    used_r, used_c, matched = set(), set(), []
    for i, j in zip(ii, jj):
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        matched.append((i, j))
    return matched


class TrackerArrays:
    """Same association logic as rknn_headraw_pipeline.Tracker, array state."""

    def __init__(self, track_high, new_track, track_low, match_iou, low_iou, max_age):
        self.next_id = 1
        self.xywh = np.zeros((0, 4), np.float32)
        self.vel = np.zeros((0, 4), np.float32)
        self.age = np.zeros((0,), np.int32)
        self.ids = np.zeros((0,), np.int64)
        self.conf = np.zeros((0,), np.float32)
        self.hi, self.new_th, self.lo = track_high, new_track, track_low
        self.miou, self.liou, self.max_age = match_iou, low_iou, max_age
        self._all = np.zeros((0,), np.int64)

    def __len__(self):
        return len(self.xywh)

    def _apply(self, rows, boxes, confs, cols):
        if not rows:
            return
        r = np.asarray(rows, np.int64)
        new = boxes[cols]
        self.vel[r] = 0.8 * self.vel[r] + 0.2 * (new - self.xywh[r])
        self.xywh[r] = new
        self.conf[r] = confs[cols]
        self.age[r] = 0

    def update(self, boxes, confs=None):
        """boxes: (D,4) xywh float32, confs: (D,) float32 (optional -> all hi)."""
        D = 0 if boxes is None else len(boxes)
        if confs is None:
            confs = np.full((D,), 1.0, np.float32)
        T = len(self.xywh)
        if T:
            self.xywh = self.xywh + self.vel
            self.age += 1
        if D:
            hi = confs >= self.hi
        else:
            hi = np.zeros((0,), bool)
        hi_boxes, hi_conf = boxes[hi], confs[hi]
        lowm = (confs >= self.lo) & ~hi if D else np.zeros((0,), bool)
        low_boxes, low_conf = (boxes[lowm], confs[lowm]) if D else (boxes, confs)

        unmatched_rows = list(range(T))
        unmatched_cols = list(range(len(hi_boxes)))
        if T and len(hi_boxes):
            M = iou_matrix(self.xywh, hi_boxes)
            matched = _assign(M, self.miou)
            if matched:
                self._apply([m[0] for m in matched], hi_boxes, hi_conf,
                            [m[1] for m in matched])
                mr = {m[0] for m in matched}
                mc = {m[1] for m in matched}
                unmatched_rows = [i for i in range(T) if i not in mr]
                unmatched_cols = [j for j in range(len(hi_boxes)) if j not in mc]

        if unmatched_rows and len(low_boxes):
            rem = np.asarray(unmatched_rows, np.int64)
            M2 = iou_matrix(self.xywh[rem], low_boxes)
            matched2 = _assign(M2, self.liou)
            if matched2:
                self._apply([int(rem[m[0]]) for m in matched2], low_boxes, low_conf,
                            [m[1] for m in matched2])

        if unmatched_cols:
            born = [j for j in unmatched_cols if hi_conf[j] >= self.new_th]
            if born:
                n = len(born)
                self.xywh = np.concatenate([self.xywh, hi_boxes[born]], 0)
                self.vel = np.concatenate([self.vel, np.zeros((n, 4), np.float32)], 0)
                self.age = np.concatenate([self.age, np.zeros(n, np.int32)])
                self.conf = np.concatenate([self.conf, hi_conf[born]], 0)
                self.ids = np.concatenate([self.ids,
                                           np.arange(self.next_id, self.next_id + n)])
                self.next_id += n

        if len(self.age):
            live = self.age <= self.max_age
            if not live.all():
                self.xywh, self.vel = self.xywh[live], self.vel[live]
                self.age, self.conf, self.ids = self.age[live], self.conf[live], self.ids[live]
        return self.xywh
