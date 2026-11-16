# -*- coding: utf-8 -*-
"""plot_arch2.py — 微调流程图 + GPT-2 结构示意 -> assets/g0_arch.png"""
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
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


def box(ax, x, y, w, h, title, sub, color, fill, tsize=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.5, edgecolor=color, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
            fontsize=tsize, weight="bold", color=color, zorder=3)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
            fontsize=8.6, color="#5b6472", zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color="#8a93a0", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, linewidth=1.5,
                                 color=color, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # ---------------- 上排: 三段流程 ----------------
    W, H, Y = 3.5, 1.55, 4.05
    xs = [0.35, 4.45, 8.55]

    box(ax, xs[0], Y, W, H, "① 下载预训练模型",
        "Hugging Face 镜像 uer/gpt2-chinese-poem\n12 层 · 768 维 · 1.03 亿参数 · 405 MB",
        "#2f6f9f", "#eef4f9")
    box(ax, xs[1], Y, W, H, "② 在《全唐诗》上继续微调",
        "11,047 首训练 / 960 首验证\nlr 5e-5 · 5 轮 · RTX 5060 上 4.9 分钟",
        "#c0642f", "#fbf1e9")
    box(ax, xs[2], Y, W, H, "③ 采样生成",
        "给定开头「春风」\n温度 / top-p 采样，逐字续写",
        "#3f8f6f", "#edf6f1")

    for i in range(2):
        arrow(ax, xs[i] + W + 0.05, Y + H / 2, xs[i + 1] - 0.05, Y + H / 2)

    ax.text(0.35, 5.95, "整体流程", fontsize=12.5, weight="bold", color="#26313d")

    # ---------------- 下排: GPT-2 内部结构 ----------------
    ax.text(0.35, 3.35, "单个样本在网络里的流向", fontsize=12.5, weight="bold", color="#26313d")

    W2, H2, Y2 = 2.05, 1.42, 1.10
    xs2 = [0.35, 2.85, 5.35, 8.15, 10.65]

    box(ax, xs2[0], Y2, W2, H2, "输入 token",
        "「春」「风」…\n(B, T)", "#2f6f9f", "#eef4f9", 10.8)
    box(ax, xs2[1], Y2, W2, H2, "词嵌入 + 位置编码",
        "22557 × 768\n加可学习位置向量", "#2f6f9f", "#eef4f9", 10.8)
    box(ax, xs2[2], Y2, 2.5, H2, "12 × Transformer Block",
        "多头自注意力（因果掩码）\n+ 前馈网络 + 残差", "#c0642f", "#fbf1e9", 10.8)
    box(ax, xs2[3], Y2, W2, H2, "线性输出层",
        "768 → 22557\n每个字的打分", "#c0642f", "#fbf1e9", 10.8)
    box(ax, xs2[4], Y2, 1.55, H2, "Softmax",
        "下一个字的\n概率分布", "#3f8f6f", "#edf6f1", 10.8)

    arrow(ax, xs2[0] + W2 + 0.05, Y2 + H2 / 2, xs2[1] - 0.05, Y2 + H2 / 2)
    arrow(ax, xs2[1] + W2 + 0.05, Y2 + H2 / 2, xs2[2] - 0.05, Y2 + H2 / 2)
    arrow(ax, xs2[2] + 2.5 + 0.05, Y2 + H2 / 2, xs2[3] - 0.05, Y2 + H2 / 2)
    arrow(ax, xs2[3] + W2 + 0.05, Y2 + H2 / 2, xs2[4] - 0.05, Y2 + H2 / 2)

    # 参数量标注
    for x, w, p in [(xs2[0], W2, "—"), (xs2[1], W2, "17.3 M"),
                    (xs2[2], 2.5, "85.1 M"), (xs2[3], W2, "17.3 M"),
                    (xs2[4], 1.55, "—")]:
        ax.text(x + w / 2, Y2 - 0.26, p, ha="center", va="center",
                fontsize=9, color="#9aa3b0")

    # 说明
    ax.text(0.35, 0.16,
            "微调时只替换训练目标：全部 1.03 亿参数都参与更新，但学习率压到 5e-5（从零训练用 3e-3），"
            "让模型在原有权重附近小幅调整，而不是推倒重来。",
            fontsize=9.2, color="#5b6472")

    plt.tight_layout()
    p = os.path.join(ASSETS, "g0_arch.png")
    plt.savefig(p, dpi=175, bbox_inches="tight")
    plt.close()
    print("saved", p)


if __name__ == "__main__":
    main()
