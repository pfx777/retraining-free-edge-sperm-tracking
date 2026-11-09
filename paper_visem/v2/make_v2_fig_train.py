#!/usr/bin/env python3
"""Training curves of the delivery recipe (v2 draft, chapter 5 / 5.2).

Three panels, all read from the Ultralytics run logs in data/train_runs:

  (a) training loss (box + cls + dfl) against epoch
  (b) validation mAP@0.5
  (c) validation mAP@0.5:0.95

Each model is drawn as the mean over its random seeds with a min-max band.
Nothing is recomputed or smoothed; the only processing is averaging seeds.

    python make_v2_fig_train.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RUNS = ROOT / "data" / "train_runs"

BLUE, GREY = "#1f6feb", "#8a8f98"

# run log -> reported group (the base model keeps two complete seeds, ours four)
BASELINE = ["stock_yolov8n_448", "stock_yolov8n_448_s7b"]
OURS = ["v8n_gctx448", "v8n_gctx448_s7", "v8n_gctx448_s123", "v8n_gctx448_s2024"]

TEXT = {
    "en": {
        "panels": [
            "(a) training loss (box + cls + dfl)",
            "(b) validation mAP@0.5",
            "(c) validation mAP@0.5:0.95",
        ],
        "xlabel": "epoch",
        "ylabel_loss": "loss",
        "l_base": "baseline stock YOLOv8n",
        "l_ours": "ours (GatedCtx)",
        "out": "fig_v2_07_train",
    },
    "zh": {
        "panels": [
            "(a) 训练损失（box + cls + dfl）",
            "(b) 验证集 mAP@0.5",
            "(c) 验证集 mAP@0.5:0.95",
        ],
        "xlabel": "epoch",
        "ylabel_loss": "损失",
        "l_base": "基线 stock YOLOv8n",
        "l_ours": "本文模型（GatedCtx）",
        "out": "fig_v2_07_train_zh",
    },
}


def read_run(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with (RUNS / f"{name}.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    loss = np.array([float(r["train/box_loss"]) + float(r["train/cls_loss"])
                     + float(r["train/dfl_loss"]) for r in rows])
    m50 = np.array([float(r["metrics/mAP50(B)"]) for r in rows])
    m95 = np.array([float(r["metrics/mAP50-95(B)"]) for r in rows])
    return np.arange(1, len(rows) + 1), loss, m50, m95


def stack(names: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
    series = [read_run(n) for n in names]
    n = min(len(c) for _, c, _, _ in series)
    epochs = series[0][0][:n]
    curves = [np.vstack([s[i][:n] for s in series]) for i in (1, 2, 3)]
    return epochs, curves


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
        "axes.unicode_minus": False, "font.size": 9, "mathtext.fontset": "stix",
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    ep_b, base = stack(BASELINE)
    ep_o, ours = stack(OURS)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3))
    for ax, idx, ylabel in zip(axes, (0, 1, 2),
                               (t["ylabel_loss"], "mAP@0.5", "mAP@0.5:0.95")):
        for ep, curves, colour, label, off in ((ep_b, base[idx], GREY, t["l_base"], (6, -2, "left")),
                                               (ep_o, ours[idx], BLUE, t["l_ours"], (-5, 6, "right"))):
            mid = curves.mean(axis=0)
            ax.plot(ep, mid, "-", color=colour, linewidth=1.5, label=label, zorder=3)
            if len(curves) > 1:
                ax.fill_between(ep, curves.min(axis=0), curves.max(axis=0),
                                color=colour, alpha=0.16, linewidth=0, zorder=2)
            ax.scatter([ep[-1]], [mid[-1]], s=18, color=colour, zorder=4)
            ax.annotate(f"{mid[-1]:.4f}" if idx else f"{mid[-1]:.2f}",
                        (ep[-1], mid[-1]), textcoords="offset points",
                        xytext=off[:2], ha=off[2], fontsize=7.4, color=colour)
        ax.set_xlabel(t["xlabel"], fontsize=8.2)
        ax.set_ylabel(ylabel, fontsize=8.2)
        ax.set_title(t["panels"][idx], loc="left", fontsize=8.8, fontweight="bold")
        ax.tick_params(labelsize=7.6)
        ax.set_xlim(0, 34)
    axes[0].legend(fontsize=7.8, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, t["out"])


def main() -> int:
    for lang in ("en", "zh"):
        figure(lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
