# -*- coding: utf-8 -*-
"""plot_finetune.py — 微调过程可视化
产出: assets/g1_loss.png  损失曲线与学习率调度
      assets/g2_samples.png  生成能力随训练演进
"""
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
OUT = os.path.join(ROOT, "outputs")
os.makedirs(ASSETS, exist_ok=True)

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


def load(tag="ft"):
    with open(os.path.join(OUT, f"{tag}_metrics.json"), encoding="utf-8") as f:
        return json.load(f)


def smooth(y, k=12):
    if len(y) < k:
        return np.array(y)
    return np.convolve(y, np.ones(k) / k, mode="valid")


def fig_loss(m, tag="ft"):
    h, st = m["history"], m["steps"]
    base = m["baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) 损失曲线
    ax = axes[0]
    ep = [x["epoch"] for x in h]
    steps = [s["step"] for s in st]
    loss = [s["loss"] for s in st]
    ax.plot(steps, loss, color=C1, alpha=0.16, lw=0.7, zorder=2, label="单步训练损失")
    sm = smooth(loss, 12)
    ax.plot(range(6, 6 + len(sm)), sm, color=C1, lw=2.2, zorder=3, label="平滑后训练损失")

    # 把每轮验证损失映射到对应步数上
    spe = len(st) / len(h)
    ax.plot([e * spe for e in ep], [x["val_loss"] for x in h], "s-",
            color=C2, ms=5, lw=1.8, zorder=4, label="每轮验证损失")
    ax.axhline(base["loss"], color=GREY, ls="--", lw=1.3, zorder=2)
    ax.text(steps[-1] * 0.015, base["loss"] + 0.055,
            f"微调前基线 {base['loss']:.3f}", fontsize=9, color=GREY)

    best = m["best"]
    ax.scatter([best["epoch"] * spe], [best["val_loss"]], s=120, facecolor="none",
               edgecolor=C3, lw=2.2, zorder=6)
    ax.annotate(f"最佳 第{best['epoch']}轮\nval {best['val_loss']:.3f}",
                (best["epoch"] * spe, best["val_loss"]),
                textcoords="offset points", xytext=(14, 16),
                fontsize=8.6, color=C3,
                arrowprops=dict(arrowstyle="-", color=C3, lw=1))

    ax.set_xlabel("优化步数")
    ax.set_ylabel("交叉熵损失")
    ax.set_title("(a) 微调损失：从预训练基线继续下降", fontsize=11, pad=8)
    ax.grid(**GRID)
    ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    # (b) 学习率 + 梯度范数
    ax = axes[1]
    ax.plot(steps, [s["lr"] for s in st], color=C1, lw=1.8, zorder=3)
    ax.set_xlabel("优化步数")
    ax.set_ylabel("学习率", color=C1)
    ax.tick_params(axis="y", labelcolor=C1)
    ax.set_title("(b) 学习率预热与余弦衰减", fontsize=11, pad=8)
    ax.grid(**GRID)

    ax2 = ax.twinx()
    ax2.plot(steps, [s["grad_norm"] for s in st], color=C2, alpha=0.45, lw=0.9, zorder=2)
    ax2.set_ylabel("梯度范数", color=C2)
    ax2.tick_params(axis="y", labelcolor=C2)
    ax2.spines["top"].set_visible(False)

    peak = max(s["lr"] for s in st)
    ax.text(0.56, 0.93, f"峰值学习率 {peak:.1e}\n共 {len(st)} 步",
            transform=ax.transAxes, fontsize=8.8, color="#5b6472",
            va="top", ha="left")

    plt.tight_layout()
    p = os.path.join(ASSETS, "g1_loss.png")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", p)


def wrap_ext(s, width=26):
    """按标点/长度切行, 用于在图中显示整首诗"""
    out, cur = [], ""
    for ch in s:
        cur += ch
        if ch in "。！？" and len(cur) >= width:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def fig_samples(m, tag="ft"):
    samples = m["samples"]
    n = len(samples)
    # 轮次不多就全画出来, 多了就均匀挑 6 个
    if n <= 6:
        picks = samples
    else:
        idx = [round(i * (n - 1) / 5) for i in range(6)]
        picks = [samples[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12.4, 1.05 * len(picks) + 1.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(picks) + 0.6)
    ax.axis("off")

    for i, s in enumerate(picks):
        y = len(picks) - i - 0.35
        ax.add_patch(plt.Rectangle((0.02, y - 0.42), 9.96, 0.84,
                                   facecolor="#f7f9fb" if i % 2 == 0 else "white",
                                   edgecolor="#e2e6ec", lw=0.9, zorder=1))
        ax.text(0.16, y, f"第 {s['epoch']} 轮", fontsize=10.5, weight="bold",
                color=C1, va="center", zorder=2)
        all_lines = wrap_ext(s["春风"], 28)
        body = all_lines[:2]
        txt = "\n".join(body) if body else "(空)"
        if len(all_lines) > 2:
            txt += "…"
        ax.text(1.45, y, txt, fontsize=10.6, color="#26313d", va="center",
                linespacing=1.55, zorder=2)

    ax.text(0.16, len(picks) + 0.18, "开头「春风」", fontsize=9.5, color=GREY)
    ax.set_title("同一个开头，不同微调阶段的续写结果", fontsize=13, weight="bold", pad=14)
    plt.tight_layout()
    p = os.path.join(ASSETS, "g2_samples.png")
    plt.savefig(p, dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", p)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "ft"
    m = load(tag)
    fig_loss(m, tag)
    fig_samples(m, tag)
