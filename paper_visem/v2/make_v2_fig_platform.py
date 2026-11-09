#!/usr/bin/env python3
"""Figure: the deployment platform (photo of the board + a short fact card).

The manuscript builder scales every figure to the text-column width, so the
figure is composed with a wide aspect: the board photo on the left and the
software-stack facts that section 4.1 states on the right.

    python make_v2_fig_platform.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
ASSETS = FIG / "v2_assets"
PHOTO = ASSETS / "board_photo.jpg"

INK, GREY, BLUE = "#1b1b1b", "#6b7280", "#1f6feb"

TEXT = {
    "en": {
        "label": "board used in this work",
        "head": "RK3568",
        "lines": [
            "RKNN INT8 runtime",
            "NPU: detector backbone + neck",
            "CPU: DFL decoding + NMS",
        ],
        "out": "fig_v2_05_platform",
    },
    "zh": {
        "label": "本文使用的开发板",
        "head": "RK3568",
        "lines": [
            "RKNN INT8 推理",
            "NPU：检测器主干与颈部",
            "CPU：DFL 解码与 NMS",
        ],
        "out": "fig_v2_05_platform_zh",
    },
}


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def figure(lang: str) -> None:
    t = TEXT[lang]
    plt.rcParams.update({
        "font.family": ("SimSun" if lang == "zh" else "serif"),
        "font.serif": (["Times New Roman", "SimSun", "Microsoft YaHei"]
                       if lang == "zh" else ["Times New Roman"]),
        "axes.unicode_minus": False, "font.size": 8, "mathtext.fontset": "stix",
    })
    img = Image.open(PHOTO)

    fig = plt.figure(figsize=(3.5, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 120)
    ax.axis("off")

    pw = 92.0
    ph = pw * img.height / img.width
    py = 116.0 - ph
    ax.imshow(img, extent=(4.0, 4.0 + pw, py, py + ph), aspect="auto", zorder=2)
    ax.add_patch(Rectangle((4.0, py), pw, ph, facecolor="none",
                           edgecolor="#9aa0a6", linewidth=0.6, zorder=3))
    ax.text(4.0 + pw / 2, py - 3.2, t["label"], ha="center", va="center",
            fontsize=6.0, color=GREY)

    ax.text(4.0, 28.0, t["head"], ha="left", va="top", fontsize=8.0,
            fontweight="bold", color=BLUE)
    yy = 20.0
    for ln in t["lines"]:
        ax.text(4.0, yy, "• " + ln, ha="left", va="top", fontsize=6.2, color=INK)
        yy -= 7.2
    save(fig, t["out"])


def main() -> int:
    if not PHOTO.exists():
        raise SystemExit(f"missing {PHOTO}")
    for lang in ("en", "zh"):
        figure(lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
