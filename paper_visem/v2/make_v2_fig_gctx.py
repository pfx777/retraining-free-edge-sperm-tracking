#!/usr/bin/env python3
"""Figure 2 of the v2 draft: the GatedCtx block, drawn in the style of the
reference figure (network context on the left, expanded detail on the right).

All labels are English.  Nothing here is measured: the figure only depicts the
forward pass already written as Eq. (1) in the manuscript.

    python make_v2_fig_gctx.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BLUE = "#1f6feb"
GREEN = "#2f9e44"
GREY = "#8a8f98"
FILL_BLUE = "#e8f0fe"
FILL_GREEN = "#eaf6ec"
FILL_GREY = "#f2f4f7"
TENSOR_EDGE, TENSOR_FILL = "#3b6ea5", "#cfe0f5"


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"   # keep text editable in SVG
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def box(ax, x, y, w, h, text, fc=FILL_GREY, ec=GREY, fs=6.6, ls="-", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=0.8",
                                linewidth=lw, facecolor=fc, edgecolor=ec,
                                linestyle=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.35, color="#1b1b1b")


def arrow(ax, x1, y1, x2, y2, color="#444", lw=0.9, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=8, linewidth=lw, color=color,
                                 linestyle=ls))


def circle_plus(ax, x, y, r=1.7, sym="+"):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor="#444",
                        linewidth=0.9, zorder=4))
    label = r"$+$" if sym == "+" else sym
    ax.text(x, y, label, ha="center", va="center", fontsize=7.6, zorder=5)


def tensor(ax, x, y, w=4.4, h=4.4, n=3, dx=0.9, dy=0.9, z=3):
    """Stacked squares, the usual way papers draw a feature map."""
    for k in range(n):
        ax.add_patch(FancyBboxPatch((x + dx * k, y + dy * k), w, h,
                                    boxstyle="square,pad=0",
                                    linewidth=0.8, facecolor=TENSOR_FILL,
                                    edgecolor=TENSOR_EDGE, zorder=z + k))


def figure():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(-4, 123)
    ax.set_ylim(-1, 47)
    ax.axis("off")

    # ---------------- left: where the block sits in the network -------------
    layers = [
        (3.0, "neck layer (l-1)", FILL_GREY, GREY),
        (12.0, "P3/8 output x", FILL_GREY, GREY),
        (21.0, "GatedCtx (l)", FILL_GREEN, GREEN),
        (30.0, "detect head (l+1)", FILL_GREY, GREY),
    ]
    for y, text, fc, ec in layers:
        box(ax, 2.0, y, 15.0, 6.0, text, fc=fc, ec=ec, fs=6.8)
    for y in (9.0, 18.0, 27.0):
        arrow(ax, 9.5, y, 9.5, y + 2.6)

    # zoom-out trapezoid from the highlighted layer to the detail panel
    arrow(ax, 17.2, 20.6, 19.8, 3.0, color=BLUE, ls=(0, (3, 2)), style="-", lw=0.8)
    arrow(ax, 17.2, 27.4, 19.8, 41.0, color=BLUE, ls=(0, (3, 2)), style="-", lw=0.8)

    # ---------------- right: the expanded block -----------------------------
    ax.add_patch(FancyBboxPatch((19.8, 1.6), 97.0, 39.8,
                                boxstyle="round,pad=0.4,rounding_size=1.4",
                                linewidth=0.9, facecolor="#fbfcfe", edgecolor=BLUE))

    # input tensor
    tensor(ax, 22.5, 20.0, w=4.2, h=4.2)
    ax.text(25.6, 18.0, "x\n64×56×56", ha="center", va="top", fontsize=6.6,
            linespacing=1.35)

    # upper branch: global context
    box(ax, 32.5, 27.0, 35.0, 10.0, "", fc="#f6fbf7", ec=GREEN,
        ls=(0, (3, 2)), lw=0.9)
    ax.text(33.8, 35.4, "Global-context branch", ha="left", va="center",
            fontsize=6.6, color=GREEN, fontweight="bold")
    box(ax, 34.0, 28.6, 7.6, 5.2, "GAP", fc=FILL_GREEN, ec=GREEN, fs=6.4)
    box(ax, 42.6, 28.6, 8.6, 5.2, "1×1 Conv\n64→8", fc=FILL_GREEN, ec=GREEN, fs=6.0)
    box(ax, 52.2, 28.6, 8.6, 5.2, "1×1 Conv\n8→64", fc=FILL_GREEN, ec=GREEN, fs=6.0)
    ax.text(64.8, 31.2, "ctx\n64×1×1", ha="center", va="center", fontsize=5.8,
            linespacing=1.3)
    for x1, x2 in ((41.6, 42.6), (51.2, 52.2)):
        arrow(ax, x1, 31.2, x2, 31.2)

    # lower branch: spatial
    box(ax, 32.5, 9.6, 33.5, 10.0, "", fc="#f7faff", ec=BLUE,
        ls=(0, (3, 2)), lw=0.9)
    ax.text(33.8, 18.0, "Spatial branch", ha="left", va="center",
            fontsize=6.6, color=BLUE, fontweight="bold")
    box(ax, 34.0, 11.2, 13.6, 5.2, "Depthwise 3×3\nConv 64→64", fc=FILL_BLUE,
        ec=BLUE, fs=6.0)
    ax.text(53.0, 13.8, "spatial(x)\n64×56×56", ha="center", va="center",
            fontsize=6.2, linespacing=1.3)

    # x fans out to the two branches
    arrow(ax, 28.6, 24.6, 32.2, 30.5)
    arrow(ax, 28.6, 21.6, 32.2, 15.0)
    # both branches converge on the sum
    arrow(ax, 67.8, 31.2, 69.0, 24.0)
    arrow(ax, 66.2, 13.8, 69.0, 20.4)

    circle_plus(ax, 70.6, 22.2)
    arrow(ax, 72.4, 22.2, 74.6, 22.2)
    box(ax, 74.8, 19.2, 8.4, 6.0, "Sigmoid", fc="#ffffff", ec="#444", fs=6.6)
    arrow(ax, 83.4, 22.2, 86.0, 22.2)
    circle_plus(ax, 87.8, 22.2, sym=r"$\otimes$")
    arrow(ax, 89.6, 22.2, 91.4, 22.2)
    box(ax, 91.6, 19.2, 10.4, 6.0, "1×1 Conv\n64→64", fc=FILL_BLUE, ec=BLUE, fs=6.0)
    arrow(ax, 102.2, 22.2, 104.0, 22.2)
    circle_plus(ax, 105.8, 22.2)
    arrow(ax, 107.6, 22.2, 110.4, 22.2)
    tensor(ax, 110.6, 20.0, w=4.2, h=4.2)
    ax.text(113.6, 18.0, "y\n64×56×56", ha="center", va="top", fontsize=6.6,
            linespacing=1.35)

    # tap of x into the gate (channel-wise product)
    arrow(ax, 25.5, 26.2, 25.5, 38.6, color="#666", style="-", lw=0.8)
    arrow(ax, 25.5, 38.6, 87.8, 38.6, color="#666", style="-", lw=0.8)
    arrow(ax, 87.8, 38.6, 87.8, 24.2, color="#666", lw=0.8)
    ax.text(56.0, 38.6, "x  (channel-wise product)", ha="center", va="center",
            fontsize=7.2, color="#333", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    # tap of x into the residual sum
    arrow(ax, 22.4, 21.0, 20.9, 21.0, color="#666", style="-", lw=0.8)
    arrow(ax, 20.9, 21.0, 20.9, 5.4, color="#666", style="-", lw=0.8)
    arrow(ax, 20.9, 5.4, 105.8, 5.4, color="#666", style="-", lw=0.8)
    arrow(ax, 105.8, 5.4, 105.8, 20.2, color="#666", lw=0.8)
    ax.text(66.0, 5.4, "x  (residual / skip connection)", ha="center",
            va="center", fontsize=7.2, color="#333", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    ax.text(68.0, 2.9,
            "y = x + Conv1×1( x $\\odot$ $\\sigma$( ctx + spatial(x) ) )        "
            "All operators are NPU-native; 6,096 parameters at 64 channels",
            ha="center", fontsize=6.4, color="#444")
    save(fig, "fig_v2_02_gctx")


def main() -> int:
    figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
