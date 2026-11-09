#!/usr/bin/env python3
"""Figure 3: the detection branches before and after export-time pruning.

Style reference: the "backbone ladder + coloured detection heads per level"
variant figure (YOLOv8 / YOLOv8s / proposed).  Panel (a) is the baseline with
three heads, panel (b) is the delivered model with the P3 head only.

"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

INK, GREY, LINE = "#1b1b1b", "#7a8087", "#5a6068"
BLUE, BLUE_EDGE, BLUE_FILL = "#1f6feb", "#9dbef4", "#f4f8ff"
GREEN, GREEN_EDGE, GREEN_FILL = "#2f9e44", "#8ac48f", "#f3faf4"
HEAD = {"P3": "#8fc98f", "P4": "#c9a6d9", "P5": "#f2c07a"}
HEAD_EDGE = {"P3": "#5f9e5f", "P4": "#9a76ad", "P5": "#cf9a4e"}
GONE, GONE_EDGE = "#f2f3f5", "#ccd2d9"

T = {
    "en": {
        "a": "(a) baseline YOLOv8n",
        "b": "(b) this work",
        "ladder": ["Input", "P1", "P2", "P3", "P4", "P5"],
        "gate": "GatedCtx",
        "gone": "removed at export",
        "shapes": {"P3": "56×56", "P4": "28×28", "P5": "14×14"},
        "heads": "detection heads",
        "stat_a": "params 3.02 M    FLOPs 4.05 G@448    ONNX 12.1 MB",
        "stat_b": "params 1.62 M (−46%)    FLOPs 3.04 G (−25%)    ONNX 6.5 MB",
        "out": "fig_v2_03_prune",
    },
    "zh": {
        "a": "(a) 基线 YOLOv8n",
        "b": "(b) 本文模型",
        "ladder": ["输入", "P1", "P2", "P3", "P4", "P5"],
        "gate": "GatedCtx",
        "gone": "导出期删除",
        "shapes": {"P3": "56×56", "P4": "28×28", "P5": "14×14"},
        "heads": "检测头",
        "stat_a": "参数 3.02 M    FLOPs 4.05 G@448    ONNX 12.1 MB",
        "stat_b": "参数 1.62 M（−46%）    FLOPs 3.04 G（−25%）    ONNX 6.5 MB",
        "out": "fig_v2_03_prune_zh",
    },
}


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def node(ax, x, y, w, h, text, fs=6.6, edge=LINE, face="white", z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.16,rounding_size=0.8",
                                linewidth=0.9, facecolor=face, edgecolor=edge,
                                zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=z + 1)


def head(ax, x, y, w, h, level, shape, grey=False, z=4):
    face = GONE if grey else HEAD[level]
    edge = GONE_EDGE if grey else HEAD_EDGE[level]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.16,rounding_size=0.8",
                                linewidth=0.9, facecolor=face, edgecolor=edge,
                                zorder=z))
    ax.text(x + w + 1.8, y + h / 2, shape, ha="left", va="center",
            fontsize=6.6, color=(GREY if grey else INK), zorder=z + 3,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    if grey:
        ax.plot([x + 0.2, x + w - 0.2], [y + h + 0.6, y - 0.4],
                color="#d1495b", linewidth=1.3, zorder=z + 2)


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=0.9, z=2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=8, linewidth=lw, color=color,
                                 zorder=z))


def ladder(ax, t, x0, ys):
    """Vertical backbone ladder: Input, P1 … P5 with downward arrows."""
    for i, name in enumerate(t["ladder"]):
        node(ax, x0, ys[i], 12.0, 4.4, name)
        if i:
            arrow(ax, x0 + 6.0, ys[i - 1], x0 + 6.0, ys[i] + 4.4)


def figure(lang: str) -> None:
    t = T[lang]
    plt.rcParams.update({
        "font.family": ("SimSun" if lang == "zh" else "serif"),
        "font.serif": (["Times New Roman", "SimSun", "Microsoft YaHei"]
                       if lang == "zh" else ["Times New Roman"]),
        "axes.unicode_minus": False, "font.size": 8, "mathtext.fontset": "stix",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    fig, ax = plt.subplots(figsize=(3.5, 3.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")

    xs = [2.0, 15.0, 28.0, 41.0, 54.0, 67.0]

    def compact_row(label: str, y: float, greyed: bool) -> None:
        ax.text(1.0, y + 11.0, label, ha="left", va="center", fontsize=7.4,
                fontweight="bold", color=INK)
        for i, name in enumerate(t["ladder"]):
            node(ax, xs[i], y, 10.0, 5.0, name, fs=5.8)
            if i:
                arrow(ax, xs[i - 1] + 10.0, y + 2.5, xs[i] - 0.2, y + 2.5)
        for level, x in zip(("P3", "P4", "P5"), xs[3:]):
            removed = greyed and level != "P3"
            arrow(ax, x + 5.0, y - 0.4, x + 5.0, y - 4.4,
                  color=(GONE_EDGE if removed else LINE), lw=0.7)
            face = GONE if removed else HEAD[level]
            edge = GONE_EDGE if removed else HEAD_EDGE[level]
            ax.add_patch(FancyBboxPatch(
                (x, y - 8.3), 10.0, 3.8,
                boxstyle="round,pad=0.12,rounding_size=0.7",
                linewidth=0.9, facecolor=face, edgecolor=edge, zorder=4))
            ax.text(x + 5.0, y - 6.4, level, ha="center", va="center",
                    fontsize=5.5, color=(GREY if removed else INK), zorder=5)
            if removed:
                ax.plot([x + 0.5, x + 9.5], [y - 8.1, y - 4.7],
                        color="#d1495b", linewidth=1.1, zorder=6)
        if greyed:
            ax.text(50.0, 1.0, t["gone"], ha="center", va="center",
                    fontsize=5.8, color=GREY)

    compact_row(t["a"], 45.0, False)
    ax.plot([1.0, 99.0], [33.5, 33.5], color="#d5d9de", linewidth=0.8)
    compact_row(t["b"], 15.0, True)

    save(fig, t["out"])


def main() -> int:
    for lang in ("en", "zh"):
        figure(lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
