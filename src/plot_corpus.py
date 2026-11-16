# -*- coding: utf-8 -*-
"""plot_corpus.py — 语料统计可视化 -> assets/fig1_corpus.png, fig2_zipf.png"""
import json
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
os.makedirs(ASSETS, exist_ok=True)

# ---- 中文字体 ----
for _f in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=_f).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#c8ccd4"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

C1, C2, C3 = "#2f6f9f", "#c0642f", "#3f8f6f"
GRID = dict(color="#e8eaee", linewidth=0.8, zorder=0)


def fig_corpus(s):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))

    # (a) 句长分布
    ax = axes[0, 0]
    d = s["line_len_dist"]
    keys = [int(k) for k in d]
    vals = [d[str(k)] for k in keys]
    keep = [(k, v) for k, v in zip(keys, vals) if v > 5 and k <= 30]
    ks, vs = zip(*keep)
    bars = ax.bar(ks, vs, color=C1, width=0.75, zorder=3)
    for k, v in zip(ks, vs):
        if v > 8000:
            ax.text(k, v + 900, f"{v:,}", ha="center", fontsize=8, color=C1)
    ax.set_xlabel("单句字符数（含标点）")
    ax.set_ylabel("句子数")
    ax.set_title("(a) 句长分布：五言(12)与七言(16)占绝对多数", fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)
    ax.set_xticks(range(6, 21, 2))
    ax.set_xlim(6, 20)
    ax.annotate("五言诗", xy=(12, 49966), xytext=(14.5, 38000), fontsize=9,
                color=C2, arrowprops=dict(arrowstyle="->", color=C2, lw=1))
    ax.annotate("七言诗", xy=(16, 10984), xytext=(19.5, 22000), fontsize=9,
                color=C3, arrowprops=dict(arrowstyle="->", color=C3, lw=1))

    # (b) 每首诗句数
    ax = axes[0, 1]
    d = s["poem_len_dist"]
    ks = [k for k in range(1, 17)]
    vs = [d.get(str(k), 0) for k in ks]
    ax.bar(ks, vs, color=C2, width=0.7, zorder=3)
    ax.set_xlabel("每首诗句数")
    ax.set_ylabel("诗歌数")
    ax.set_title("(b) 每首诗的句数：四句/八句为主", fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)
    ax.set_xticks(ks)

    # (c) 高频字（排除标点，只统计汉字）
    ax = axes[1, 0]
    top = [(c, n) for c, n in s["top_chars"] if "\u4e00" <= c <= "\u9fff"][:20]
    chars = [c for c, _ in top]
    cnt = [n for _, n in top]
    ax.bar(range(len(chars)), cnt, color=C1, width=0.72, zorder=3)
    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels(chars, fontsize=9)
    ax.set_xlim(-0.8, len(chars) - 0.2)
    ax.set_ylabel("出现次数")
    ax.set_title("(c) 出现频次最高的 20 个字", fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)

    # (d) 关键数字
    ax = axes[1, 1]
    ax.axis("off")
    rows = [("诗歌总数", f"{s['num_poems']:,} 首"),
            ("诗句总数", f"{s['num_lines']:,} 行"),
            ("语料字符数", f"{s['num_chars']:,}"),
            ("字符表大小", f"{s['vocab_size']:,}"),
            ("平均每句", f"{s['num_chars'] / s['num_lines']:.1f} 字")]
    ax.text(0.02, 0.92, "语料概要", fontsize=13, weight="bold", color="#26313d")
    for i, (k, v) in enumerate(rows):
        y = 0.78 - i * 0.155
        ax.text(0.04, y, k, fontsize=11, color="#5b6472", va="center")
        ax.text(0.52, y, v, fontsize=12, color=C1, weight="bold", va="center")
        ax.plot([0.02, 0.92], [y - 0.055, y - 0.055], color="#eef0f3", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.suptitle("《全唐诗》语料统计分析", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(ASSETS, "fig1_corpus.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)


def fig_zipf(s):
    """字符频次的双对数分布——验证 Zipf 定律（只统计汉字）"""
    from collections import Counter
    with open(os.path.join(ROOT, "data", "poetry.txt"), encoding="utf-8") as f:
        text = f.read()
    cnt = Counter(c for c in text if "\u4e00" <= c <= "\u9fff")
    freq = np.array([v for _, v in cnt.most_common(800)], dtype=float)
    rank = np.arange(1, len(freq) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(rank, freq, "o", ms=2.6, color=C1, alpha=0.65, label="实测频次")
    # 拟合幂律
    k = (rank >= 5) & (rank <= 300)
    coef = np.polyfit(np.log(rank[k]), np.log(freq[k]), 1)
    ax.loglog(rank, np.exp(coef[1]) * rank ** coef[0], "--",
              color=C2, lw=1.6, label=f"幂律拟合 α≈{-coef[0]:.2f}")
    ax.set_xlabel("字频排名（log）")
    ax.set_ylabel("出现次数（log）")
    ax.set_title("字符频次服从 Zipf 定律：少数高频字支撑大部分文本", fontsize=11)
    ax.grid(True, which="both", color="#eef0f3", lw=0.7)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    p = os.path.join(ASSETS, "fig2_zipf.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)


if __name__ == "__main__":
    with open(os.path.join(ROOT, "data", "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    fig_corpus(stats)
    fig_zipf(stats)
