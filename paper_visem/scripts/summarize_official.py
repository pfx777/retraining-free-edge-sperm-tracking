#!/usr/bin/env python3
"""Collate the four same-protocol evaluations into one table (JSON + Markdown)."""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
ROWS = [
    ("ours baseline YOLOv8n-448 FP32", "official_compare_baseline_yp32.json"),
    ("ours E26 (GatedCtx) 448 FP32", "official_compare_E26_yp32.json"),
    ("official YOLOv5n-640 (c0 only)", "official_compare_yolov5n.json"),
    ("official YOLOv5x-640 (c0 only)", "official_compare_yolov5x.json"),
]


def main() -> int:
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "—"

    out = {}
    lines = ["| 检测器 | 不校正 | κ 规则 λ=0.5 | κ 规则 λ=1.0 | 动态 λ | oracle | κ 拟合值 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, fn in ROWS:
        p = DATA / fn
        if not p.exists():
            print("missing", fn)
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        dyn = j.get("mean_dynamic_k2", j.get("mean_dynamic_k4"))
        kap = j.get("kappa_fitted", 3.066)
        out[name] = {
            "none": j.get("mean_mota_none"),
            "kappa_l05": j.get("mean_mota_lambda05"),
            "kappa_l10": j.get("mean_mota_lambda10"),
            "dynamic_k2": j.get("mean_dynamic_k2"),
            "dynamic_k4": j.get("mean_dynamic_k4"),
            "oracle": j.get("mean_mota_oracle"),
            "kappa_fitted": kap,
            "sessions": j.get("kappa_fit_sessions"),
            "per_session": {v: {k: j["val"][v][k] for k in
                                ("s_hat", "q90", "mota_none", "mota_lambda05",
                                 "mota_lambda10", "oracle_scale", "mota_oracle")}
                            for v in j.get("val", {})},
        }
        lines.append(
            f"| {name} | {fmt(j.get('mean_mota_none'))} | {fmt(j.get('mean_mota_lambda05'))} | "
            f"{fmt(j.get('mean_mota_lambda10'))} | {fmt(dyn)} | {fmt(j.get('mean_mota_oracle'))} | "
            f"{fmt(kap)} |")

    (DATA / "official_compare_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    header = [
        "# 官方 VISEM 基线与本方法：同一协议对照（val = 52/54/60/82，k=1，统一 class-0 MOTA）",
        "",
        "协议：class-0 GT、我们的板端追踪器、IoU≥0.5 贪心匹配、ID 切换计给 GT、gap≤14 插值、逐段输出框尺度。",
        "官方权重是 3 类（sperm / cluster / small_or_pinhead），此处只保留 class 0 以匹配本文 GT 口径；",
        "官方权重为 FP32、img 640、50 epoch；本方法为 448、30 epoch。κ 在 16 个训练段上重拟合（线索项来自 kappa_outfit.log）。",
        "",
    ]
    (DATA / "official_compare_summary.md").write_text(
        "\n".join(header + lines + [""]), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
