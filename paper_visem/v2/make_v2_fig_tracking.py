#!/usr/bin/env python3
"""Figure 8 of the v2 draft: qualitative tracking results on one real session.

One colour convention throughout, so the panels can be read side by side:
    green  = class-0 ground truth
    grey   = baseline INT8 detections
    blue   = delivered INT8 detections / track boxes
    orange = track trails

Session 82 (one of the four unseen videos).  The frame shown in (a)-(d) is the
one with the largest honest gain of the delivered model over the baseline; the
frame in (e) is the one with the most unmatched tracks.

Inputs (all local): datasets/visem_tracking/mot_gt_c0/82.txt,
.tmp/base_4sess/dets_82, .tmp/pruned_4sess/dets_gctx_82,
.tmp/trk_pruned/trk_pruned_82, VISEM_Tracking_Train_v4/Train/82/82.mp4

    python make_v2_fig_tracking.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

CODE = Path(r"D:\code")
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

FW, FH = 640.0, 480.0
SESSION = "82"
MAX_FRAME = 200

GREEN, GREY, BLUE, ORANGE = "#2f9e44", "#8a8f98", "#1f6feb", "#e08a1e"

T = {
    "en": {
        "a": "(a) class-0 ground truth ({n} targets)",
        "b": "(b) baseline INT8: {hit} hit / {miss} missed",
        "c": "(c) delivered INT8: {hit} hit / {miss} missed",
        "d": "(d) delivered tracks: {n} identities\n(orange = trails)",
        "e": "(e) failure mode: {n} tracks, {fp} unmatched\n(frame {fr})",
        "f": "(f) duplicate ids on one cell",
        "out": "fig_v2_08_detection",
        "out2": "fig_v2_09_tracking",
    },
    "zh": {
        "a": "(a) class-0 真值（{n} 个目标）",
        "b": "(b) 基线 INT8：命中 {hit} / 漏 {miss}",
        "c": "(c) 交付件 INT8：命中 {hit} / 漏 {miss}",
        "d": "(d) 交付件追踪：{n} 条身份\n（橙 = 尾迹）",
        "e": "(e) 失败模式：{n} 条轨迹，{fp} 条未匹配\n（第 {fr} 帧）",
        "f": "(f) 同一条精子上的重复 ID",
        "out": "fig_v2_08_detection_zh",
        "out2": "fig_v2_09_tracking_zh",
    },
}


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def read_gt(v: str):
    out = defaultdict(list)
    for line in (CODE / "datasets" / "visem_tracking" / "mot_gt_c0" / f"{v}.txt").read_text().splitlines():
        q = line.replace(",", " ").split()
        if len(q) >= 6:
            out[int(float(q[0]))].append((float(q[2]), float(q[3]), float(q[4]), float(q[5])))
    return out


def read_dets(path: Path):
    out = defaultdict(list)
    for line in path.read_text().splitlines():
        q = line.split()
        if len(q) >= 6:
            out[int(float(q[0]))].append((float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5])))
    return out


def read_tracks(v: str):
    d = CODE / ".tmp" / "trk_pruned" / f"trk_pruned_{v}"
    out = defaultdict(list)
    for f in d.glob("*.txt"):
        fr = int(f.stem)
        for line in f.read_text().splitlines():
            q = line.split()
            if len(q) >= 6:
                out[fr].append((int(float(q[5])), float(q[1]), float(q[2]), float(q[3]), float(q[4])))
    return out


def iou(a, b) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax1 + a[2], bx1 + b[2]), min(ay1 + a[3], by1 + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def matched(gt_frame, boxes) -> tuple[int, int]:
    pairs = sorted(((iou(g, b), i, j) for i, g in enumerate(gt_frame)
                    for j, b in enumerate(boxes)), reverse=True)
    used_g, used_b = set(), set()
    for s, i, j in pairs:
        if s >= 0.5 and i not in used_g and j not in used_b:
            used_g.add(i)
            used_b.add(j)
    return len(used_g), len(gt_frame) - len(used_g)


def frame_image(v: str, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(CODE / "VISEM_Tracking_Train_v4" / "Train" / v / f"{v}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {v} frame {frame}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw(ax, img, gt=None, boxes=None, box_color=GREY, tracks=None, trail=None,
         lw=0.9, labels=False):
    ax.imshow(img)
    ax.set_xlim(0, FW)
    ax.set_ylim(FH, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#888")
    for g in (gt or []):
        ax.add_patch(Rectangle((g[0] * FW - g[2] * FW / 2, g[1] * FH - g[3] * FH / 2),
                               g[2] * FW, g[3] * FH, fill=False, edgecolor=GREEN, lw=1.0))
    for b in (boxes or []):
        ax.add_patch(Rectangle((b[0] * FW - b[2] * FW / 2, b[1] * FH - b[3] * FH / 2),
                               b[2] * FW, b[3] * FH, fill=False, edgecolor=box_color, lw=lw))
    if tracks is not None and trail:
        for tid, pts in trail.items():
            ax.plot([p[0] * FW for p in pts], [p[1] * FH for p in pts], "-",
                    color=ORANGE, lw=0.9, alpha=0.9)
    if labels and tracks:
        for t in tracks:
            ax.text(t[1] * FW + 2, t[2] * FH - 2, f"{t[0]}", color=BLUE,
                    fontsize=6.0, weight="bold")


def figure(lang: str) -> None:
    t = T[lang]
    plt.rcParams.update({
        "font.family": ("SimSun" if lang == "zh" else "serif"),
        "font.serif": (["Times New Roman", "SimSun", "Microsoft YaHei"]
                       if lang == "zh" else ["Times New Roman"]),
        "axes.unicode_minus": False, "font.size": 8, "mathtext.fontset": "stix",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    scales = json.loads((ROOT / "data" / "results.json").read_text(encoding="utf-8"))
    dyn = scales["board_int8_full_length"]["scales_dynamic"]
    base_scale = dict(zip(["82", "60", "54", "52"],
                          scales["board_int8_full_length"]["baseline_own_four_clue"]["scales"]))[SESSION]
    ours_scale = dyn[SESSION]

    gt_all = read_gt(SESSION)
    base = read_dets(CODE / ".tmp" / "base_4sess" / f"dets_{SESSION}" / "detections.txt")
    ours = read_dets(CODE / ".tmp" / "pruned_4sess" / f"dets_gctx_{SESSION}" / "detections.txt")
    tracks = read_tracks(SESSION)

    def scaled(rows, s):
        return [(cx, cy, w * s, h * s, sc) for cx, cy, w, h, sc in rows]

    best_gain, best_fail = (-99, 1), (-99, 1)
    for fr in range(1, MAX_FRAME + 1):
        g = gt_all.get(fr, [])
        if g:
            _, miss_b = matched(g, scaled(base.get(fr, []), base_scale))
            _, miss_o = matched(g, scaled(ours.get(fr, []), ours_scale))
            if miss_b - miss_o > best_gain[0]:
                best_gain = (miss_b - miss_o, fr)
        tr = tracks.get(fr, [])
        if tr:
            m, _ = matched(g, [(x[1], x[2], x[3], x[4]) for x in tr])
            if len(tr) - m > best_fail[0]:
                best_fail = (len(tr) - m, fr)
    f_gain, f_fail = best_gain[1], best_fail[1]
    print(f"session {SESSION}: gain frame {f_gain}, failure frame {f_fail}")

    img_g, img_f = frame_image(SESSION, f_gain), frame_image(SESSION, f_fail)
    g = gt_all.get(f_gain, [])
    b_boxes = scaled(base.get(f_gain, []), base_scale)
    o_boxes = scaled(ours.get(f_gain, []), ours_scale)
    hit_b, miss_b = matched(g, b_boxes)
    hit_o, miss_o = matched(g, o_boxes)
    tr_g = tracks.get(f_gain, [])
    tr_f = tracks.get(f_fail, [])
    trail_g, trail_f = {}, {}
    for fr in range(max(1, f_gain - 20), f_gain + 1):
        for x in tracks.get(fr, []):
            trail_g.setdefault(x[0], []).append((x[1], x[2]))
    for fr in range(max(1, f_fail - 20), f_fail + 1):
        for x in tracks.get(fr, []):
            trail_f.setdefault(x[0], []).append((x[1], x[2]))
    tr_f_matched, _ = matched(gt_all.get(f_fail, []),
                              [(x[1], x[2], x[3], x[4]) for x in tr_f])

    # ---- figure 8: detection, baseline vs delivered ------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1))
    plt.subplots_adjust(wspace=0.06)
    draw(axes[0], img_g, gt=g)
    axes[0].set_title(t["a"].format(n=len(g)), loc="left", fontsize=7.6,
                      fontweight="bold")
    draw(axes[1], img_g, gt=g, boxes=b_boxes, box_color=GREY)
    axes[1].set_title(t["b"].format(hit=hit_b, miss=miss_b), loc="left",
                      fontsize=7.6, fontweight="bold")
    draw(axes[2], img_g, gt=g, boxes=o_boxes, box_color=BLUE, lw=1.0)
    axes[2].set_title(t["c"].format(hit=hit_o, miss=miss_o), loc="left",
                      fontsize=7.6, fontweight="bold")
    save(fig, t["out"])

    # ---- figure 9: tracking, trajectories and one failure ------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1))
    plt.subplots_adjust(wspace=0.06)
    draw(axes[0], img_g, gt=g, tracks=tr_g, trail=trail_g, labels=True)
    axes[0].set_title(t["d"].format(n=len(tr_g)), loc="left", fontsize=7.6,
                      fontweight="bold")
    draw(axes[1], img_f, gt=gt_all.get(f_fail, []), tracks=tr_f, trail=trail_f,
         labels=True)
    axes[1].set_title(t["e"].format(n=len(tr_f), fp=len(tr_f) - tr_f_matched,
                                    fr=f_fail),
                      loc="left", fontsize=7.6, fontweight="bold")
    # (f) zoom on the most overlapping pair of tracks in the failure frame
    best = (0.0, None, None, None)
    for i in range(len(tr_f)):
        for j in range(i + 1, len(tr_f)):
            a = tr_f[i][1:5]
            b = tr_f[j][1:5]
            v = iou(a, b)
            if v > best[0]:
                best = (v, i, j, (a, b))
    if best[3] is not None:
        a, b = best[3]
        cx = 0.5 * (a[0] + b[0]) * FW
        cy = 0.5 * (a[1] + b[1]) * FH
        half = 70
        x0 = int(max(0, min(FW - 2 * half, cx - half)))
        y0 = int(max(0, min(FH - 2 * half, cy - half)))
        zoom = img_f[y0:y0 + 2 * half, x0:x0 + 2 * half]
        axes[2].imshow(zoom)
        axes[2].set_xticks([]); axes[2].set_yticks([])
        axes[2].set_xlim(0, 2 * half)
        axes[2].set_ylim(2 * half, 0)
        for (tid, bx, by, bw, bh) in (tr_f[best[1]], tr_f[best[2]]):
            px, py = bx * FW - x0, by * FH - y0
            axes[2].add_patch(Rectangle((px - bw * FW / 2, py - bh * FH / 2),
                                        bw * FW, bh * FH, fill=False,
                                        edgecolor=BLUE, lw=1.2))
            axes[2].text(px + 2, py - 3, f"{tid}", color=BLUE, fontsize=7.0,
                         weight="bold")
        for spine in axes[2].spines.values():
            spine.set_linewidth(0.6); spine.set_color("#888")
        axes[2].set_title(t["f"], loc="left", fontsize=7.6, fontweight="bold")
    save(fig, t["out2"])


def main() -> int:
    figure("en")
    figure("zh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
