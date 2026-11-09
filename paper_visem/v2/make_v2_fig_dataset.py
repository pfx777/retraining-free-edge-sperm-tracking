#!/usr/bin/env python3
"""Figure 7 of the v2 draft: what the VISEM-Tracking data looks like.

(a) raw frame, (b) the same frame with class-0 annotations, (c) a 4x zoom on a
single spermatozoon (the red box in (a) marks the crop), (d) annotated targets
per session (16 training + 4 unseen), (e) distribution of annotation width.

Panels (a)-(c) use real frames of session 82 (crops taken from
../figures/fig7_qualitative.png); panels (d)-(e) read
../data/v2_dataset_stats.json produced by dataset_stats.py.

    python make_v2_fig_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
ASSETS = FIG / "v2_assets"
STATS = json.loads((ROOT / "data" / "v2_dataset_stats.json").read_text(encoding="utf-8"))

BLUE, ORANGE, GREY = "#1f6feb", "#e08a1e", "#8a8f98"

# crop of a_raw.png used for the zoom panel: (left, top, size) in pixels
ZOOM = (246, 325, 96)

T = {
    "en": {
        "t_a": "(a) raw frame, 640×480",
        "t_b": "(b) class-0 annotations",
        "t_c": "(c) 4× zoom: one sperm cell (box ≈ 22 px)",
        "t_d": "(d) annotated targets per session",
        "t_e": "(e) annotation width",
        "d_x": "session",
        "d_y": "targets",
        "e_x": "box width (px)",
        "e_y": "annotations",
        "leg_train": "16 training sessions",
        "leg_val": "4 unseen sessions",
        "d_note": "20 sessions · 29,370 frames · 1,121 annotated targets · 612,377 GT rows",
        "e_note": "median {med:g} px  (5th–95th {p5:g}–{p95:g} px)",
        "out": "fig_v2_07_dataset",
    },
    "zh": {
        "t_a": "(a) 原始帧 640×480",
        "t_b": "(b) class-0 逐帧标注",
        "t_c": "(c) 4× 放大：单条精子（标注框约 22 px）",
        "t_d": "(d) 每段视频的标注目标数",
        "t_e": "(e) 标注框宽度分布",
        "d_x": "视频段",
        "d_y": "目标数",
        "e_x": "框宽 (px)",
        "e_y": "标注数",
        "leg_train": "16 个训练段",
        "leg_val": "4 段未见视频",
        "d_note": "20 段 · 29,370 帧 · 1,121 条标注目标 · 612,377 行 GT",
        "e_note": "中位 {med:g} px（5–95 分位 {p5:g}–{p95:g} px）",
        "out": "fig_v2_07_dataset_zh",
    },
}


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def figure(lang: str):
    t = T[lang]
    plt.rcParams.update({
        "font.family": ("SimSun" if lang == "zh" else "serif"),
        "font.serif": (["Times New Roman", "SimSun", "Microsoft YaHei"]
                       if lang == "zh" else ["Times New Roman"]),
        "axes.unicode_minus": False, "font.size": 8, "mathtext.fontset": "stix",
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    # A flatter canvas keeps this figure (image + caption) inside one page of the
    # reading PDF; the photo row is width-limited, so it is unaffected.
    fig = plt.figure(figsize=(7.2, 4.1))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.30, wspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    raw = Image.open(ASSETS / "a_raw.png").convert("RGB")
    gt = Image.open(ASSETS / "b_gt.png").convert("RGB")

    # ---- (a) raw frame, with the zoom region marked -----------------------
    ax_a.imshow(raw)
    ax_a.set_xticks([]); ax_a.set_yticks([]); ax_a.grid(False)
    x0, y0, s = ZOOM
    ax_a.add_patch(Rectangle((x0, y0), s, s, fill=False, edgecolor="#d1495b",
                             linewidth=1.4))
    ax_a.set_title(t["t_a"], loc="left", fontsize=7.8, fontweight="bold")

    # ---- (b) annotations --------------------------------------------------
    ax_b.imshow(gt)
    ax_b.set_xticks([]); ax_b.set_yticks([]); ax_b.grid(False)
    ax_b.set_title(t["t_b"], loc="left", fontsize=7.8, fontweight="bold")

    # ---- (c) zoom ---------------------------------------------------------
    crop = raw.crop((x0, y0, x0 + s, y0 + s)).resize((s * 4, s * 4), Image.NEAREST)
    ax_c.imshow(crop)
    ax_c.set_xticks([]); ax_c.set_yticks([]); ax_c.grid(False)
    ax_c.set_title(t["t_c"], loc="left", fontsize=7.8, fontweight="bold")

    # ---- (d) targets per session ------------------------------------------
    tr = STATS["train"]
    va = STATS["val"]
    order = tr + va
    xs = np.arange(len(order))
    vals = [r["targets"] for r in order]
    colors = [GREY] * len(tr) + [ORANGE] * len(va)
    ax_d.bar(xs, vals, color=colors, width=0.72, zorder=3)
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels([r["session"] for r in order], fontsize=6.0)
    ax_d.set_xlabel(t["d_x"], fontsize=7.0)
    ax_d.set_ylabel(t["d_y"], fontsize=7.0)
    ax_d.tick_params(labelsize=6.4)
    ax_d.set_title(t["t_d"], loc="left", fontsize=7.8, fontweight="bold")
    ax_d.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREY),
                         plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
                labels=[t["leg_train"], t["leg_val"]], fontsize=6.2,
                frameon=False, loc="upper left")
    # ---- (e) annotation width --------------------------------------------
    counts = np.array(STATS["box_w_hist"]["counts"], dtype=float)
    edges = np.array(STATS["box_w_hist"]["edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_e.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=BLUE,
             zorder=3)
    med = STATS["summary"]["box_w_px_median"]
    ax_e.axvline(med, color="#d1495b", linewidth=1.2, linestyle="--", zorder=4)
    ax_e.set_xlabel(t["e_x"], fontsize=7.0)
    ax_e.set_ylabel(t["e_y"], fontsize=7.0)
    ax_e.tick_params(labelsize=6.4)
    ax_e.set_xlim(0, 60)
    ax_e.set_title(t["t_e"], loc="left", fontsize=7.8, fontweight="bold")
    ax_e.text(0.99, 0.86, t["e_note"].format(med=STATS["summary"]["box_w_px_median"], p5=STATS["summary"]["box_w_px_p05"], p95=STATS["summary"]["box_w_px_p95"]),
              transform=ax_e.transAxes, ha="right", fontsize=6.2, color="#555")
    save(fig, t["out"])


def main() -> int:
    figure("en")
    figure("zh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
