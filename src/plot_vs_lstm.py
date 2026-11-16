# -*- coding: utf-8 -*-
"""plot_vs_lstm.py — 从零训练 vs 微调预训练模型

关键点: 两个模型的词表大小差 3.6 倍, 困惑度不能直接比较。
    LSTM  : 6229 个字, ln(V) = 8.74
    GPT-2 : 22557 个字, ln(V) = 10.02
公平的做法是看「归一化熵」 = 交叉熵 / ln(V), 即模型还剩下多少不确定度。

产出: assets/g5_vs_lstm.png
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

C1, C2, C3 = "#2f6f9f", "#c0642f", "#3f8f6f"
GREY = "#8b95a3"
GRID = dict(color="#e8eaee", linewidth=0.8, zorder=0)


def jload(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def collect():
    rows = []

    m = jload("lstm_metrics.json")
    best = min(m["history"], key=lambda x: x["val_loss"])
    rows.append({
        "name": "字符级\nLSTM",
        "vocab": 6229,
        "loss": best["val_loss"],
        "ppl": best["val_ppl"],
        "acc": best["val_acc"],
        "params": m["params"]["total"],
        "time": m["total_time"],
        "color": GREY,
    })

    b = jload("base_metrics.json")
    rows.append({
        "name": "GPT-2\n微调前",
        "vocab": 22557,
        "loss": b["loss"],
        "ppl": b["ppl"],
        "acc": b["acc"],
        "params": b["params"],
        "time": None,
        "color": "#b8c0cc",
    })

    f = jload("ft_metrics.json")
    rows.append({
        "name": "GPT-2\n微调后",
        "vocab": 22557,
        "loss": f["history"][-1]["val_loss"],
        "ppl": f["history"][-1]["val_ppl"],
        "acc": f["history"][-1]["val_acc"],
        "params": f["params"]["total"],
        "time": f["total_time"],
        "color": C1,
    })

    for r in rows:
        r["norm"] = r["loss"] / math.log(r["vocab"])
    return rows


def main():
    rows = collect()
    for r in rows:
        print(f"{r['name'].replace(chr(10),' '):16s} ppl {r['ppl']:9.1f} "
              f"归一化熵 {r['norm']:.3f} acc {r['acc']*100:.1f}% "
              f"参数 {r['params']/1e6:.2f}M")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = np.arange(len(rows))
    names = [r["name"] for r in rows]

    # (a) 困惑度
    ax = axes[0]
    bars = ax.bar(x, [r["ppl"] for r in rows], width=0.55,
                  color=[r["color"] for r in rows], zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("验证困惑度（对数轴）")
    ax.set_title("(a) 验证困惑度：三者词表不同，不能直接比", fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, r["ppl"] * 1.12,
                f"{r['ppl']:.1f}", ha="center", fontsize=10, weight="bold",
                color="#26313d")
        ax.text(b.get_x() + b.get_width() / 2, r["ppl"] * 0.62,
                f"词表 {r['vocab']}", ha="center", fontsize=8,
                color="white" if r["color"] != "#b8c0cc" else "#5b6472")
    ax.set_ylim(30, 900)

    # (b) 归一化熵
    ax = axes[1]
    bars = ax.bar(x, [r["norm"] for r in rows], width=0.55,
                  color=[r["color"] for r in rows], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("归一化熵  =  交叉熵 / ln(词表大小)")
    ax.set_title("(b) 归一化熵：剔除词表大小影响后的对比",
                 fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, r["norm"] + 0.006,
                f"{r['norm']:.3f}", ha="center", fontsize=10, weight="bold",
                color="#26313d")
    ax.set_ylim(0, 1.06)
    ax.axhline(1.0, color="#d9534f", ls=":", lw=1.2, zorder=2)
    ax.text(0.5, 1.015, "1.0 = 完全随机猜测", fontsize=8.2, color="#d9534f",
            va="bottom", ha="left")

    plt.tight_layout()
    p = os.path.join(ASSETS, "g5_vs_lstm.png")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", p)


if __name__ == "__main__":
    main()
