#!/usr/bin/env python3
"""Figure 1 of the v2 draft: horizontal pipeline in the reference paper's style.

Five numbered stage panels from left to right, real VISEM-Tracking frames as
thumbnails, and colour-coded compute: NPU modules blue, CPU modules orange.
All labels are in English (figures aim at an English-language submission).

Thumbnails are cropped from ../figures/fig7_qualitative.png (session 82, frames
27 and 45) into ../figures/v2_assets/ -> a_raw.png, d_deliv.png, e_tracks.png.

    python make_v2_fig_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
ASSETS = FIG / "v2_assets"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

NPU_FILL, NPU_EDGE = "#dbe8ff", "#1f6feb"
CPU_FILL, CPU_EDGE = "#fdeccd", "#e08a1e"
PANEL_FILL = ["#f7f8fa", "#eef4ff", "#eef4ff", "#fdf6ec", "#f3f8f3"]
PANEL_EDGE = ["#c9cfd6", "#bcd3f7", "#bcd3f7", "#f0d6ab", "#c8e0c9"]

STAGE_TITLES = ["1. Input", "2. Detector", "3. Detection head", "4. Tracker", "5. Output"]


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"   # keep text editable in SVG
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def panel(ax, x, y, w, h, i):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.35,rounding_size=1.2",
                                linewidth=0.9, facecolor=PANEL_FILL[i],
                                edgecolor=PANEL_EDGE[i], linestyle=(0, (4, 2))))
    ax.text(x + w / 2, y + h + 2.2, STAGE_TITLES[i], ha="center", va="bottom",
            fontsize=8.6, fontweight="bold", color="#333333")


def block(ax, x, y, w, h, text, kind="npu", fs=6.9):
    fill, edge = (NPU_FILL, NPU_EDGE) if kind == "npu" else (CPU_FILL, CPU_EDGE)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.35,rounding_size=1.0",
                                linewidth=1.0, facecolor=fill, edgecolor=edge))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.4, color="#1b1b1b")


def thumb(ax, name, x, y, w, h, dx=0.0, dy=0.0, z=3):
    img = Image.open(ASSETS / name)
    ax.imshow(img, extent=(x + dx, x + w + dx, y + dy, y + h + dy),
              aspect="auto", zorder=z)
    ax.add_patch(FancyBboxPatch((x + dx, y + dy), w, h,
                                boxstyle="square,pad=0", linewidth=0.5,
                                facecolor="none", edgecolor="#888", zorder=z + 1))


def arrow(ax, x1, y1, x2, y2, color="#444", lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=10, linewidth=lw, color=color))


def figure():
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    w, gap, x0 = 17.4, 3.0, 0.5
    xs = [x0 + i * (w + gap) for i in range(5)]
    py, ph = 24.0, 25.0
    cx = [x + w / 2 for x in xs]

    for i, x in enumerate(xs):
        panel(ax, x, py, w, ph, i)

    # ---- 1. input -----------------------------------------------------------
    for k, (dx, dy) in enumerate([(-1.0, -1.0), (-0.5, -0.5), (0.0, 0.0)]):
        thumb(ax, "a_raw.png", xs[0] + 1.4, py + 8.6, w - 3.4, 13.4,
              dx=dx, dy=dy, z=3 + k)
    ax.text(cx[0], py + 5.6, "Microscope frames", ha="center",
            fontsize=6.9, color="#333")
    ax.text(cx[0], py + 3.8, "640×480", ha="center",
            fontsize=6.4, color="#333")

    # ---- 2. detector --------------------------------------------------------
    block(ax, xs[1] + 1.4, py + 18.6, w - 2.8, 6.4,
          "YOLOv8n\nsingle-class · INT8", kind="npu", fs=6.6)
    thumb(ax, "d_deliv.png", xs[1] + 1.4, py + 4.0, w - 3.4, 12.8)
    ax.text(cx[1], py + 2.0, "Boxes", ha="center",
            fontsize=6.9, color="#333")

    # ---- 3. detection head --------------------------------------------------
    block(ax, xs[2] + 1.4, py + 18.6, w - 2.8, 6.4,
          "P3/8 + GatedCtx\n6,096 params", kind="npu", fs=6.6)
    block(ax, xs[2] + 1.4, py + 10.4, w - 2.8, 6.4,
          "P3-only head\nsingle output", kind="npu", fs=6.6)
    # kept / removed detection branches, as a compact strip instead of prose
    for j, (name, keep) in enumerate((("P3", True), ("P4", False), ("P5", False))):
        bx = xs[2] + 2.0 + j * 4.7
        ax.add_patch(FancyBboxPatch((bx, py + 2.6), 4.0, 3.4,
                                    boxstyle="round,pad=0.12,rounding_size=0.8",
                                    linewidth=0.9,
                                    facecolor=(NPU_FILL if keep else "#f1f2f4"),
                                    edgecolor=(NPU_EDGE if keep else "#b9bec6"),
                                    zorder=3))
        ax.text(bx + 2.0, py + 4.3, name, ha="center", va="center",
                fontsize=6.4, color=("#1b1b1b" if keep else "#8a8f98"), zorder=4)
        if not keep:
            ax.plot([bx + 0.3, bx + 3.7], [py + 3.0, py + 5.6],
                    color="#8a8f98", linewidth=0.9, zorder=4)

    # ---- 4. tracker ---------------------------------------------------------
    block(ax, xs[3] + 1.4, py + 18.6, w - 2.8, 6.4,
          "Association tracker\nevery k frames", kind="cpu", fs=6.6)
    thumb(ax, "e_tracks.png", xs[3] + 1.4, py + 4.0, w - 3.4, 12.8)
    ax.text(cx[3], py + 2.0, "Tracks + IDs", ha="center",
            fontsize=6.9, color="#333")

    # ---- 5. output ----------------------------------------------------------
    for k, (dx, dy) in enumerate([(0.0, 0.0), (0.8, 0.8)]):
        thumb(ax, "e_tracks.png", xs[4] + 1.6, py + 8.8, w - 4.4, 12.0,
              dx=dx, dy=dy, z=3 + k)
    ax.text(cx[4], py + 6.0, "Tracks + kinematics", ha="center",
            fontsize=6.9, color="#333")
    ax.text(cx[4], py + 4.2, "MOTA / HOTA / IDF1", ha="center",
            fontsize=6.9, color="#333")

    # ---- inter-stage arrows -------------------------------------------------
    for i in range(4):
        ax1 = xs[i] + w + 0.5
        arrow(ax, ax1, py + ph / 2, ax1 + gap - 1.0, py + ph / 2)

    # ---- callout 2: export-side changes ------------------------------------
    block(ax, xs[2] - 1.5, 4.6, w + 3.0, 8.8,
          "[1] GatedCtx replaces attention\n"
          "dead branches removed (−46%)",
          kind="npu", fs=5.7)
    arrow(ax, cx[2], py - 0.6, cx[2], 13.6, color=NPU_EDGE)

    # ---- callout 1: inference-side change ----------------------------------
    block(ax, xs[3] + 0.5, 4.6, w + 3.0, 8.8,
          "[2] per-session scale (cues + q90)\n"
          "applied to the output boxes",
          kind="cpu", fs=5.7)
    arrow(ax, cx[3], py - 0.6, cx[3], 13.6, color=CPU_EDGE)

    ax.text(50, 54.6, "Blue: NPU      Orange: CPU",
            ha="center", fontsize=7.0, color="#555")
    save(fig, "fig_v2_01_pipeline")


def main() -> int:
    missing = [n for n in ("a_raw.png", "d_deliv.png", "e_tracks.png")
               if not (ASSETS / n).exists()]
    if missing:
        raise SystemExit(f"missing assets {missing}; run the crop step first")
    figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
