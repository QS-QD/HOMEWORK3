# -*- coding: utf-8 -*-
"""plot_arch.py — 模型结构示意图 -> assets/fig3_arch.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

for _f in ["C:/Windows/Fonts/msyh.ttc"]:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=_f).get_name()]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, title, sub, color, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1.5, edgecolor=color, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center",
            fontsize=11.5, weight="bold", color=color, zorder=3)
    ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
            fontsize=8.5, color="#5b6472", zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#8a93a0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.4,
                                 color=color, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(1.0, 5.0)
    ax.axis("off")

    W, H, Y = 1.85, 1.25, 2.35
    xs = [0.35, 2.75, 5.15, 7.55, 9.95]

    box(ax, xs[0], Y, W, H, "输入字符", "如「春」\n(B, T)", "#2f6f9f", "#eef4f9")
    box(ax, xs[1], Y, W, H, "Embedding", "6229 × 128\n(B, T, 128)", "#2f6f9f", "#eef4f9")
    box(ax, xs[2], Y, W, H, "LSTM ×1", "hidden 256\n(B, T, 256)", "#c0642f", "#fbf1e9")
    box(ax, xs[3], Y, W, H, "Linear", "256 → 6229\n(B, T, 6229)", "#c0642f", "#fbf1e9")
    box(ax, xs[4], Y, W, H, "Softmax", "下一个字的\n概率分布", "#3f8f6f", "#edf6f1")

    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + W + 0.03, Y + H / 2, xs[i + 1] - 0.03, Y + H / 2)

    # 参数量标注
    params = ["—", "0.80 M", "0.39 M", "1.59 M", "—"]
    for x, p in zip(xs, params):
        ax.text(x + W / 2, Y - 0.30, p, ha="center", va="center",
                fontsize=9, color="#9aa3b0")

    # 循环箭头: LSTM 隐状态自循环
    ax.add_patch(FancyBboxPatch((xs[2] + 0.55, Y + H + 0.18), 0.75, 0.42,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                linewidth=1.2, edgecolor="#c0642f",
                                facecolor="white", zorder=2))
    ax.text(xs[2] + 0.925, Y + H + 0.39, "h, c", ha="center", va="center",
            fontsize=8.5, color="#c0642f", zorder=3)
    ax.add_patch(FancyArrowPatch((xs[2] + 0.55, Y + H + 0.39), (xs[2] + 0.30, Y + H / 2 + 0.2),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.2, color="#c0642f",
                                 connectionstyle="arc3,rad=0.3", zorder=1))
    ax.add_patch(FancyArrowPatch((xs[2] + 1.30, Y + H / 2 + 0.2), (xs[2] + 1.30, Y + H + 0.39),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.2, color="#c0642f",
                                 connectionstyle="arc3,rad=0.3", zorder=1))
    ax.text(xs[2] + W / 2, Y + H + 0.75, "隐状态跨时间步传递", ha="center",
            fontsize=8.5, color="#c0642f")

    # 标题与总参数
    ax.text(0.35, 4.55, "字符级 LSTM 语言模型结构（总参数 2.78 M）",
            fontsize=13.5, weight="bold", color="#26313d")
    ax.text(0.35, 4.10, "把「预测下一个字」当作 6229 分类问题：每读一个字，输出整个字符表上的概率分布",
            fontsize=9.5, color="#5b6472")

    ax.text(0.35, 1.75, "训练目标：交叉熵损失  L = −(1/T)·Σ log P(y_t | y_<t)",
            fontsize=10, color="#26313d")
    ax.text(0.35, 1.35, "推理方式：每步取概率最高的字（或按温度采样）拼回输入，循环生成整首诗",
            fontsize=10, color="#26313d")

    plt.tight_layout()
    p = os.path.join(ASSETS, "fig3_arch.png")
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print("saved", p)


if __name__ == "__main__":
    main()
