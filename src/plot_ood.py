# -*- coding: utf-8 -*-
"""plot_ood.py — 灾难性遗忘检测结果可视化
产出: assets/g6_ood.png
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "outputs")

for _f in ["C:/Windows/Fonts/msyh.ttc"]:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=_f).get_name()]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#c8ccd4"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

C1, C2 = "#2f6f9f", "#c0642f"
GREY = "#8b95a3"
GRID = dict(color="#e8eaee", linewidth=0.8, zorder=0)


def jload(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    ood = jload("ood_metrics.json")
    base = jload("base_metrics.json")
    ft = jload("ft_metrics.json")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    # (a) 四段现代汉语上, 微调前后的困惑度
    ax = axes[0]
    rows_b = ood["models"]["base"]["rows"]
    rows_f = ood["models"]["ft_best"]["rows"]
    n = len(rows_b)
    x = np.arange(n)
    w = 0.36
    b1 = ax.bar(x - w / 2, [r["ppl"] for r in rows_b], w, color=GREY,
                zorder=3, label="微调前")
    b2 = ax.bar(x + w / 2, [r["ppl"] for r in rows_f], w, color=C1,
                zorder=3, label="微调后")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["第 1 段\n（交通）", "第 2 段\n（软件）",
                        "第 3 段\n（气象）", "第 4 段\n（教育）"], fontsize=9.5)
    ax.set_ylabel("困惑度（对数轴）")
    ax.set_title("(a) 域外现代汉语文本：微调后困惑度同样下降",
                 fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for bs, rs in [(b1, rows_b), (b2, rows_f)]:
        for b, r in zip(bs, rs):
            ax.text(b.get_x() + b.get_width() / 2, r["ppl"] * 1.15,
                    f"{r['ppl']/1000:.1f}k", ha="center", fontsize=8, color="#5b6472")
    ax.set_ylim(1000, 90000)

    # (b) 域内 / 域外 的变化幅度
    ax = axes[1]
    in_b, in_f = base["ppl"], ft["history"][-1]["val_ppl"]
    out_b = ood["models"]["base"]["mean_ppl"]
    out_f = ood["models"]["ft_best"]["mean_ppl"]
    labels = ["唐诗验证集\n（域内）", "现代汉语\n（域外）"]
    deltas = [(in_f - in_b) / in_b * 100, (out_f - out_b) / out_b * 100]
    y = np.arange(len(labels))
    bars = ax.barh(y, deltas, height=0.5, color=[C1, C2], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="#9aa3af", lw=1)
    ax.set_xlabel("困惑度变化率（%）")
    ax.set_title("(b) 两个场景都在下降，没有出现遗忘", fontsize=11, pad=8)
    ax.grid(axis="x", **GRID)
    ax.invert_yaxis()
    for b, d in zip(bars, deltas):
        ax.text(d / 2, b.get_y() + b.get_height() / 2, f"{d:.1f}%",
                va="center", ha="center", fontsize=12, weight="bold", color="white")
    ax.set_xlim(min(deltas) * 1.10, 6)

    plt.tight_layout()
    p = os.path.join(ASSETS, "g6_ood.png")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", p)


if __name__ == "__main__":
    main()
