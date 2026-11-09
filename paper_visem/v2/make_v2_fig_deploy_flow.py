#!/usr/bin/env python3
"""Figure 5 of the v2 draft: deployment flow, kept as sparse as the reference
figure it follows -- one short label per box, dashed rounded labels on the
arrows, brace-grouped training stack on the left, dashed architecture stack on
the right.  Every detail (dtype, mean/std, output count, equations) belongs to
the caption, not to the drawing.

    python make_v2_fig_deploy_flow.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

INK = "#1b1b1b"
GREEN, GREEN_FILL = "#2f9e44", "#eaf7ec"

T = {
    "en": {
        "m1": "PyTorch",
        "m2": "YOLOv8n",
        "g_train": "Training model",
        "a1": "Export + prune",
        "b_onnx": "ONNX",
        "a2": "Quantize",
        "b_q": "RKNN INT8",
        "a3": "Compile",
        "b_c": "Offline model",
        "a4": "Deploy",
        "s1": "NPU runtime API",
        "s2": "NPU driver",
        "s3": "RK3568",
        "g_sw": "Software architecture",
        "a5": "Run",
        "b_cal": "Per-session scale adaptation",
        "b_trk": "Tracking",
        "g_rt": "Runtime stage",
        "note": "green = the two steps added in this paper",
        "out": "fig_v2_05_deploy_flow",
    },
    "zh": {
        "m1": "PyTorch",
        "m2": "YOLOv8n",
        "g_train": "训练模型",
        "a1": "导出 + 剪枝",
        "b_onnx": "ONNX",
        "a2": "量化",
        "b_q": "RKNN INT8",
        "a3": "编译",
        "b_c": "离线模型",
        "a4": "部署",
        "s1": "NPU 运行时 API",
        "s2": "NPU 驱动",
        "s3": "RK3568",
        "g_sw": "软件架构",
        "a5": "运行",
        "b_cal": "尺度校准",
        "b_trk": "追踪",
        "g_rt": "运行阶段",
        "note": "绿色 = 本文新增的两步",
        "out": "fig_v2_05_deploy_flow_zh",
    },
}


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def frame(ax, x, y, w, h, ec=INK, lw=1.3, ls="-", fc="white", z=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, linestyle=ls, zorder=z))


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=lw, color=color,
                                 zorder=3))


def tag(ax, xc, yc, text, color=INK, fs=6.6):
    w = max(8.0, 0.62 * len(text) + 2.6)
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - 1.7), w, 3.4,
                                boxstyle="round,pad=0.15,rounding_size=1.7",
                                linewidth=0.9, facecolor="white",
                                edgecolor=color, linestyle=(0, (2.6, 2.0)),
                                zorder=5))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, color=color,
            zorder=6)


def figure(lang: str):
    t = T[lang]
    plt.rcParams.update({
        "font.family": ("SimSun" if lang == "zh" else "serif"),
        "font.serif": (["Times New Roman", "SimSun", "Microsoft YaHei"]
                       if lang == "zh" else ["Times New Roman"]),
        "axes.unicode_minus": False, "font.size": 8, "mathtext.fontset": "stix",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    fig, ax = plt.subplots(figsize=(3.25, 3.2))
    ax.set_xlim(10, 90)
    ax.set_ylim(0, 80)
    ax.axis("off")
    boxes = [
        (66.0, t["m2"], "white", INK),
        (51.5, t["b_onnx"], "white", INK),
        (37.0, t["b_q"], "white", INK),
        (22.5, t["s3"] + "\n" + t["b_c"], "white", INK),
        (8.0, t["b_cal"] + "\n→ " + t["b_trk"], "white", INK),
    ]
    widths = [74.0, 56.0, 56.0, 74.0, 74.0]
    for (yy, label, face, edge), width in zip(boxes, widths):
        x = 50.0 - width / 2
        frame(ax, x, yy, width, 8.2, ec=edge,
              lw=(1.3 if edge == GREEN else 1.0), fc=face)
        ax.text(50.0, yy + 4.1, label, ha="center", va="center",
                fontsize=6.8, color=edge, linespacing=1.35)

    specs = [
        (66.0, 59.7, t["a1"]),
        (51.5, 45.2, t["a2"]),
        (37.0, 30.7, t["a3"]),
        (22.5, 16.2, t["b_cal"]),
    ]
    for y1, y2, label in specs:
        colour = GREEN if label in {t["a1"], t["b_cal"]} else INK
        arrow(ax, 50.0, y1, 50.0, y2, color=colour, lw=1.0)
        ax.text(52.0, (y1 + y2) / 2, label, ha="left", va="center",
                fontsize=5.8, color=colour)
    ax.text(50.0, 2.0, t["note"], ha="center", fontsize=5.8, color=GREEN)
    save(fig, t["out"])


def main() -> int:
    figure("en")
    figure("zh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
