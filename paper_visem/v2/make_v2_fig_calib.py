#!/usr/bin/env python3
"""Figure 4: compact two-column calibration workflow.

The left side uses parallel training and inference columns so the figure fills
its vertical space. The right panel keeps the original qualitative comparison,
with numeric values and an axis shown for readability.
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

INK, GREY = "#1b1b1b", "#7a8087"
BLUE, BLUE_FILL = "#1f6feb", "#eef5ff"
GREEN, GREEN_FILL = "#2f9e44", "#eef9f0"
ORANGE, BAND = "#e08a1e", "#c2c7ce"

T = {
    "en": {
        "train": "Training  (one-off; labels available)",
        "infer": "Inference  (no labels)",
        "train_steps": [
            "training videos\nwith labels",
            "image cues + optimal scale\n$f_{mean}$, $lapvar$, $s_{opt}$",
            "least-squares fit\n$\\kappa$, $q90_{ref}$, $sd_{ref}$",
        ],
        "infer_steps": [
            "new video\nwithout labels",
            "image cues + scores\n$f_{mean}$, $lapvar$, $q90_v$",
            "scale rule\n$\\lambda_v \\rightarrow s_v \\rightarrow s_{box}$",
        ],
        "out": "one $s_{box}$ per video,\napplied to tracked boxes",
        "panel_title": "rule scale vs needed scale",
        "panel_sub": "on unseen videos",
        "legend_rule": "scale from the rule",
        "legend_need": "needed (per-session best)",
        "axis": "box scale",
        "outfile": "fig_v2_04_calib",
    },
    "zh": {
        "train": "训练（一次性；已有标注）",
        "infer": "推理（无标注）",
        "train_steps": [
            "训练视频\n及其标注",
            "图像线索 + 最优尺度\n$f_{mean}$、$lapvar$、$s_{opt}$",
            "最小二乘拟合\n$\\kappa$、$q90_{ref}$、$sd_{ref}$",
        ],
        "infer_steps": [
            "无标注\n新视频",
            "图像线索 + 检测分数\n$f_{mean}$、$lapvar$、$q90_v$",
            "尺度规则\n$\\lambda_v \\rightarrow s_v \\rightarrow s_{box}$",
        ],
        "out": "每段视频一个 $s_{box}$，\n作用于追踪输出的框",
        "panel_title": "规则尺度 vs 所需尺度",
        "panel_sub": "在未见视频上",
        "legend_rule": "规则给出的尺度",
        "legend_need": "该段需要的尺度",
        "axis": "框尺度",
        "outfile": "fig_v2_04_calib_zh",
    },
}


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def f2(value: float) -> str:
    return str(
        Decimal(repr(float(value))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )


def box(ax, x, y, w, h, text, edge, fill="white", color=INK, fs=6.7):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.25,rounding_size=1.8",
            linewidth=1.0,
            facecolor=fill,
            edgecolor=edge,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        linespacing=1.45,
        zorder=4,
    )


def arrow(ax, start, end, color=GREY, lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def figure(lang: str) -> None:
    t = T[lang]
    plt.rcParams.update(
        {
            "font.family": "SimSun" if lang == "zh" else "serif",
            "font.serif": (
                ["Times New Roman", "SimSun", "Microsoft YaHei"]
                if lang == "zh"
                else ["Times New Roman"]
            ),
            "axes.unicode_minus": False,
            "font.size": 8,
            "mathtext.fontset": "stix",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    data = json.loads((ROOT / "data" / "results.json").read_text(encoding="utf-8"))
    dyn = data["dynamic_lambda"]
    scale = dyn["per_session_scale"]
    need = dyn["required_scale_unseen"]

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.set_xlim(0, 134)
    ax.set_ylim(0, 64)
    ax.axis("off")

    ax.text(3.0, 61.4, t["train"], ha="left", va="center", fontsize=7.3,
            color=GREY, fontweight="bold")
    ax.text(46.0, 61.4, t["infer"], ha="left", va="center", fontsize=7.3,
            color=BLUE, fontweight="bold")

    train_x, infer_x = 3.0, 46.0
    w, h = 36.0, 9.2
    row_y = [45.0, 27.5, 10.0]
    train_boxes = []
    infer_boxes = []

    for i, text in enumerate(t["train_steps"]):
        is_frozen = i == 2
        train_boxes.append(
            box(
                ax,
                train_x,
                row_y[i],
                w,
                h,
                text,
                GREEN if is_frozen else GREY,
                GREEN_FILL if is_frozen else "white",
                GREEN if is_frozen else INK,
            )
        )
        if i:
            arrow(
                ax,
                (train_x + w / 2, row_y[i - 1] - 0.3),
                (train_x + w / 2, row_y[i] + h + 0.6),
                color=GREEN if is_frozen else GREY,
            )

    for i, text in enumerate(t["infer_steps"]):
        is_output = i == 2
        infer_boxes.append(
            box(
                ax,
                infer_x,
                row_y[i],
                w,
                h,
                text,
                BLUE,
                BLUE_FILL if is_output else "white",
                INK,
            )
        )
        if i:
            arrow(
                ax,
                (infer_x + w / 2, row_y[i - 1] - 0.3),
                (infer_x + w / 2, row_y[i] + h + 0.6),
                color=BLUE,
            )

    arrow(
        ax,
        (train_x + w + 0.3, row_y[2] + h / 2),
        (infer_x - 0.6, row_y[2] + h / 2),
        color=GREEN,
        lw=1.5,
    )

    ax.text(
        infer_x + w / 2,
        5.4,
        t["out"],
        ha="center",
        va="center",
        fontsize=5.5,
        color=BLUE,
        linespacing=1.25,
    )
    arrow(
        ax,
        (infer_x + w / 2, row_y[2] - 0.3),
        (infer_x + w / 2, 7.3),
        color=BLUE,
    )

    ax.add_patch(
        FancyBboxPatch(
            (92, 7.0),
            40.5,
            54.5,
            boxstyle="round,pad=0.4,rounding_size=1.8",
            linewidth=0.9,
            facecolor="white",
            edgecolor="#d5d9de",
            zorder=1,
        )
    )
    ax.text(94.5, 59.0, t["panel_title"], ha="left", va="center",
            fontsize=7.2, color=INK, fontweight="bold", zorder=2)
    ax.text(94.5, 55.8, t["panel_sub"], ha="left", va="center",
            fontsize=6.4, color=GREY, zorder=2)
    ax.plot([95.9], [52.6], "o", ms=4.8, color=BLUE, zorder=3)
    ax.text(97.0, 52.6, t["legend_rule"], ha="left", va="center",
            fontsize=6.2, color=GREY, zorder=2)
    ax.plot([95.9], [49.9], "D", ms=4.6, color=ORANGE, zorder=3)
    ax.text(97.0, 49.9, t["legend_need"], ha="left", va="center",
            fontsize=6.2, color=GREY, zorder=2)

    axa = ax.inset_axes([97.0, 13.0, 33.0, 34.0], transform=ax.transData)
    order = ["82", "52", "60", "54"]
    rows = {v: len(order) - 1 - i for i, v in enumerate(order)}
    for v in order:
        y = rows[v]
        a, b = need[v], scale[v]
        axa.plot([b, a], [y, y], "-", color=BAND, linewidth=2.0,
                 zorder=2, solid_capstyle="round")
        axa.plot([b], [y], "o", ms=5.4, color=BLUE, zorder=4)
        axa.plot([a], [y], "D", ms=5.8, mfc=ORANGE, mec="white", mew=0.7,
                 zorder=5)
        axa.annotate(
            f2(a),
            (a, y),
            textcoords="offset points",
            xytext=(0, -11 if a < b else 7),
            ha="center",
            fontsize=5.8,
            color=ORANGE,
            zorder=5,
        )
        axa.annotate(
            f2(b),
            (b, y),
            textcoords="offset points",
            xytext=(0, 7 if a < b else -11),
            ha="center",
            fontsize=5.8,
            color=BLUE,
            zorder=5,
        )
    axa.set_xlim(0.79, 1.48)
    axa.set_ylim(-0.55, len(order) - 0.45)
    axa.set_yticks([rows[v] for v in order])
    axa.set_yticklabels(["A", "B", "C", "D"], fontsize=5.9)
    axa.set_xticks([0.8, 1.0, 1.2, 1.4])
    axa.set_xticklabels(["0.8", "1.0", "1.2", "1.4"])
    axa.tick_params(labelsize=5.8)
    axa.tick_params(axis="y", length=0)
    axa.set_xlabel(t["axis"], fontsize=6.2)
    axa.grid(axis="y", alpha=0.16, linewidth=0.5)
    axa.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axa.spines[spine].set_visible(False)
    axa.spines["bottom"].set_color("#c2c7ce")

    save(fig, t["outfile"])


def main() -> int:
    for lang in ("en", "zh"):
        figure(lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
