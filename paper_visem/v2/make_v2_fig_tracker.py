#!/usr/bin/env python3
"""Figure 10 of the v2 draft: tracker replacement under k-frame sampling.

Same board INT8 detections, same frames, same per-session box scale; only the
tracker changes.  Two bars per tracker (k=1, k=4) plus a star for the fixed
variant of the three public trackers at k=4.

All numbers come from data/results.json -> tracker_comparison.

    python make_v2_fig_tracker.py
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
D = json.loads((ROOT / "data" / "results.json").read_text(encoding="utf-8"))["tracker_comparison"]

BLUE, ORANGE, GREY = "#1f6feb", "#e08a1e", "#6b7280"

T = {
    "en": {
        "names": ["ours", "ByteTrack", "BoT-SORT", "OC-SORT"],
        "k1": "$k=1$ (13.2 FPS)",
        "k4": "$k=4$ (50.6 FPS)",
        "ylabel": "MOTA",
        "title": "Tracker replacement",
        "note": "★ = public tracker with the unconfirmed-track bug fixed\n"
                "(kept instead of deleted on an unmatched call)",
        "out": "fig_v2_10_trackers",
    },
    "zh": {
        "names": ["本文追踪器", "ByteTrack", "BoT-SORT", "OC-SORT"],
        "k1": "$k=1$（13.2 FPS）",
        "k4": "$k=4$（50.6 FPS）",
        "ylabel": "MOTA",
        "title": "追踪器替换",
        "note": "★ = 修掉『未确认轨迹被删』这一条之后的公开追踪器",
        "out": "fig_v2_10_trackers_zh",
    },
}


def save(fig, name: str) -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def figure(lang: str) -> None:
    t = T[lang]
    plt.rcParams.update({
        "font.family": ("SimSun" if lang == "zh" else "serif"),
        "font.serif": (["Times New Roman", "SimSun", "Microsoft YaHei"]
                       if lang == "zh" else ["Times New Roman"]),
        "axes.unicode_minus": False, "font.size": 8, "mathtext.fontset": "stix",
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    k1 = D["k1_mean_mota"]
    k4 = D["k4_mean_mota"]
    keys = ["ours", "bytetrack_default", "botsort_default", "ocsort_default"]
    fixed = {"bytetrack_default": k4["bytetrack_unconfirmed_kept_4sess"],
             "botsort_default": k4["botsort_unconfirmed_kept_4sess"],
             "ocsort_default": k4["ocsort_minhits1"]}
    v1 = [k1[k] for k in keys]
    v4 = [k4[k] for k in keys]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    x = np.arange(len(keys))
    w = 0.34
    b1 = ax.bar(x - w / 2, v1, width=w, color=BLUE, edgecolor="white",
                linewidth=0.6, label=t["k1"], zorder=3)
    b2 = ax.bar(x + w / 2, v4, width=w, color=ORANGE, edgecolor="white",
                linewidth=0.6, label=t["k4"], zorder=3)
    for i, k in enumerate(keys):
        if k in fixed:
            ax.scatter([x[i] + w / 2], [fixed[k]], marker="*", s=70,
                       color="#2f9e44", zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels(t["names"], fontsize=6.2, rotation=24, ha="right")
    ax.set_ylabel(t["ylabel"], fontsize=7.0)
    ax.set_ylim(0, 0.78)
    ax.tick_params(labelsize=6.6)
    ax.set_title(t["title"], loc="left", fontsize=7.8, fontweight="bold")
    ax.legend(fontsize=6.1, frameon=False, loc="upper right", ncol=1)
    save(fig, t["out"])


def main() -> int:
    figure("en")
    figure("zh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
