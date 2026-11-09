#!/usr/bin/env python3
"""Board-result figures of the v2 draft, split into two separate figures because
they belong to two different chapters:

* fig_v2_06_board_results -- what the board actually produced (章 4 边缘系统实现
  / 4.4.2): (a) the tracking confusion counts of the host replay and of the board
  run, cell by cell identical; (b) grouped bars for board latency -- three builds
  x two metrics (NPU inference / end-to-end), values printed on the bars,
  delivered-vs-baseline delta annotated above the last group;
* fig_v2_10_stride   -- MOTA against board FPS for the k-frame sampling ladder
  (章 5 实验 / 5.3.4 跨帧采样).

All numbers are read from data/results.json (delivery_verification / latency /
stride); nothing is recomputed here.

    python make_v2_fig_deploy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
D = json.loads((ROOT / "data" / "results.json").read_text(encoding="utf-8"))

BLUE, ORANGE, GREY = "#1f6feb", "#e08a1e", "#8a8f98"

T_LAT = {
    "en": {
        "bars": ["Baseline\nstock YOLOv8n", "+GatedCtx\n(unpruned)", "Ours\n(single output)"],
        "l_npu": "NPU inference",
        "l_e2e": "end-to-end",
        "panel": "Board latency on RK3568 (INT8, same measurement flow)",
        "ylabel": "latency (ms)",
        "delta": "\u22125.6% vs baseline (NPU)",
        "a_panel": "Confusion counts: host replay vs board run\n(session 82, first 200 frames, $k=4$)",
        "a_host": "host replay",
        "a_board": "board INT8 (RK3568)",
        "a_rows": ["truth present", "truth absent"],
        "a_cols": ["detected", "not detected"],
        "a_cells": ["TP", "FN", "FP"],
        "a_na": "n/a",
        "a_same": "cell by cell identical",
        "a_note": "identity switches IDSW = 16 on both sides; MOTA = 0.7301 on both sides",
        "out": "fig_v2_06_board_results",
    },
    "zh": {
        "bars": ["\u57fa\u7ebf\nstock YOLOv8n", "\u52a0\u5165 GatedCtx\n\u672a\u526a\u679d", "\u672c\u6587\u6a21\u578b\n\u5355\u8f93\u51fa"],
        "l_npu": "NPU \u63a8\u7406",
        "l_e2e": "\u7aef\u5230\u7aef",
        "panel": "RK3568 \u4e0a\u7684\u5ef6\u8fdf\u5bf9\u6bd4\uff08INT8\uff0c\u540c\u4e00\u5957\u6d4b\u91cf\u6d41\u7a0b\uff09",
        "ylabel": "\u5ef6\u8fdf (ms)",
        "delta": "\u8f83\u57fa\u7ebf -5.6%\uff08NPU\uff09",
        "a_panel": "\u6df7\u6dc6\u77e9\u9635\uff1a\u4e3b\u673a\u91cd\u653e\u4e0e\u677f\u7aef\u9010\u683c\u76f8\u540c\n\uff08\u7b2c 82 \u6bb5\u524d 200 \u5e27\uff0c$k=4$ \u91c7\u6837\uff09",
        "a_host": "\u4e3b\u673a\u91cd\u653e",
        "a_board": "\u677f\u7aef INT8\uff08RK3568\uff09",
        "a_rows": ["\u771f\u503c\u5b58\u5728", "\u771f\u503c\u4e0d\u5b58\u5728"],
        "a_cols": ["\u68c0\u51fa", "\u672a\u68c0\u51fa"],
        "a_cells": ["TP", "FN", "FP"],
        "a_na": "\u4e0d\u9002\u7528",
        "a_same": "\u9010\u683c\u76f8\u540c",
        "a_note": "\u8eab\u4efd\u5207\u6362 IDSW = 16\uff0c\u4e24\u8fb9\u76f8\u540c\uff1bMOTA \u5747\u4e3a 0.7301",
        "out": "fig_v2_06_board_results_zh",
    },
}

T_STR = {
    "en": {
        "panel": "MOTA vs. board throughput",
        "xlabel": "board FPS",
        "ylabel": "MOTA",
        "l_base": "baseline stock",
        "l_deliv": "ours",
        "note": "ours at k=4 (50.6 FPS) is still above baseline at k=1 (13.2 FPS)",
        "out": "fig_v2_10_stride",
    },
    "zh": {
        "panel": "MOTA\u2013\u5e27\u7387",
        "xlabel": "\u677f\u7aef\u5e27\u7387",
        "ylabel": "MOTA",
        "l_base": "\u57fa\u7ebf stock",
        "l_deliv": "\u672c\u6587\u6a21\u578b",
        "note": "\u672c\u6587\u6a21\u578b\u5728 k=4\uff0850.6 FPS\uff09\u4ecd\u9ad8\u4e8e\u57fa\u7ebf\u5728 k=1\uff0813.2 FPS\uff09",
        "out": "fig_v2_10_stride_zh",
    },
}


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def _rc(lang: str) -> None:
    plt.rcParams.update({
        "font.family": ("SimSun" if lang == "zh" else "serif"),
        "font.serif": (["Times New Roman", "SimSun", "Microsoft YaHei"]
                       if lang == "zh" else ["Times New Roman"]),
        "axes.unicode_minus": False, "font.size": 8, "mathtext.fontset": "stix",
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _confusion_panel(ax, t: dict) -> None:
    """(a) the tracking confusion counts, host replay next to the board run."""
    counts = D["delivery_verification"]["tracker_dump_stride4_82_200frames"]
    cc = counts["confusion_counts"]

    ax.set_xlim(0, 100)
    ax.set_ylim(28, 84)
    ax.axis("off")
    ax.set_title(t["a_panel"], loc="left", fontsize=7.8, fontweight="bold")

    cw, chh = 20.0, 13.0
    y0 = 42.0
    mats = [(12.0, t["a_host"]), (58.0, t["a_board"])]
    face = {"TP": "#e8f4ea", "FN": "#fdeee0", "FP": "#fbecec", "NA": "#f2f3f5"}
    edge = {"TP": "#6aa46f", "FN": "#d79a5b", "FP": "#c98a8a", "NA": "#c8ccd2"}
    cells = [("TP", cc["TP"]), ("FN", cc["FN"]), ("FP", cc["FP"]), ("NA", None)]
    for i, (x0, who) in enumerate(mats):
        for j, (key, val) in enumerate(cells):
            r, c = divmod(j, 2)          # row 0 = truth present
            x = x0 + c * cw
            y = y0 + (1 - r) * chh
            ax.add_patch(plt.Rectangle((x, y), cw, chh, facecolor=face[key],
                                       edgecolor=edge[key], linewidth=0.9,
                                       zorder=2))
            txt = t["a_na"] if val is None else f"{val:,}"
            ax.text(x + cw / 2, y + chh / 2 + (1.6 if val is not None else 0),
                    txt, ha="center", va="center",
                    fontsize=9.0 if val is not None else 6.4,
                    color="#1b1b1b" if val is not None else "#9aa0a6", zorder=3)
            if val is not None:
                ax.text(x + cw / 2, y + chh / 2 - 4.2, key, ha="center",
                        va="center", fontsize=5.8, color="#5b6169", zorder=3)
        for c, cl in enumerate(t["a_cols"]):
            ax.text(x0 + (c + 0.5) * cw, y0 + 2 * chh + 1.5, cl,
                    ha="center", va="bottom", fontsize=6.0, color="#6b7280")
        ax.text(x0 + cw, y0 + 2 * chh + 7.0, who, ha="center", va="bottom",
                fontsize=6.8, color="#1b1b1b")
        if i == 0:
            for r, rl in enumerate(t["a_rows"]):
                ax.text(x0 - 1.5, y0 + (1.5 - r) * chh, rl, ha="right",
                        va="center", fontsize=6.0, color="#6b7280")
    ax.text(55.0, y0 + chh, "\u2261", ha="center", va="center", fontsize=12,
            color="#2f9e44")
    ax.text(55.0, y0 - 4.5, t["a_same"], ha="center", va="center", fontsize=6.2,
            color="#2f9e44")
    ax.text(55.0, y0 - 11.0, t["a_note"], ha="center", va="center", fontsize=6.0,
            color="#6b7280")


def figure_board_results(lang: str) -> None:
    """Fig 7: (a) confusion counts host vs board, (b) grouped latency bars."""
    t = T_LAT[lang]
    _rc(lang)
    plt.rcParams["axes.grid"] = False

    lat = D["latency"]["ms"]
    npu = [lat["baseline"]["npu"], lat["e26"]["npu"], lat["e35"]["npu"]]
    e2e = [lat["baseline"]["full"], lat["e26"]["full"], lat["e35"]["full"]]

    fig = plt.figure(figsize=(7.2, 3.4))
    axc = fig.add_axes([0.005, 0.02, 0.50, 0.90])
    ax = fig.add_axes([0.60, 0.14, 0.385, 0.74])
    _confusion_panel(axc, t)
    plt.rcParams["axes.grid"] = True
    ax.set_axisbelow(True)
    x = np.arange(3)
    w = 0.34
    b1 = ax.bar(x - w / 2, npu, width=w, color=BLUE, edgecolor="white",
                linewidth=0.6, label=t["l_npu"], zorder=3)
    b2 = ax.bar(x + w / 2, e2e, width=w, color=ORANGE, edgecolor="white",
                linewidth=0.6, label=t["l_e2e"], zorder=3)
    for rect, v in list(zip(b1, npu)) + list(zip(b2, e2e)):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.2,
                f"{v:g}", ha="center", fontsize=6.6)
    ax.set_xticks(x)
    ax.set_xticklabels(t["bars"], fontsize=6.6)
    ax.set_ylabel(t["ylabel"], fontsize=7.0)
    ax.set_ylim(0, 92)
    ax.tick_params(labelsize=6.6)
    ax.set_title(t["panel"], loc="left", fontsize=7.8, fontweight="bold")
    ax.legend(fontsize=6.6, frameon=False, loc="upper left", ncol=2)
    ax.text(1.62, 84.0, t["delta"], ha="center", fontsize=6.6, color=BLUE)
    save(fig, t["out"])


def figure_stride(lang: str) -> None:
    """(b) MOTA against board FPS for the k-frame sampling ladder."""
    t = T_STR[lang]
    _rc(lang)

    st = D["stride"]
    fps = [st["fps"][k] for k in ("1", "2", "3", "4")]
    base = st["board_int8_output_scale"]["baseline"]
    deliv = st["board_int8_output_scale"]["e35"]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(fps, base, "o-", color=GREY, ms=4, linewidth=1.4, label=t["l_base"])
    ax.plot(fps, deliv, "o-", color=BLUE, ms=4, linewidth=1.4, label=t["l_deliv"])
    ax.axhline(base[0], color=GREY, linestyle=":", linewidth=1.0)
    ax.scatter([fps[0]], [base[0]], marker="*", s=70, color=GREY, zorder=5)
    ax.scatter([fps[3]], [deliv[3]], marker="*", s=70, color=BLUE, zorder=5)
    ax.set_xticks(fps)
    ax.set_xticklabels([f"{v:g}" for v in fps])
    ax.set_xlabel(t["xlabel"], fontsize=7.0)
    ax.set_ylabel(t["ylabel"], fontsize=7.0)
    ax.set_title(t["panel"], loc="left", fontsize=7.8, fontweight="bold")
    ax.legend(fontsize=6.2, frameon=False, loc="lower right")
    ax.tick_params(labelsize=6.6)
    fig.tight_layout()
    save(fig, t["out"])


def main() -> int:
    for lang in ("en", "zh"):
        figure_board_results(lang)
        figure_stride(lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
