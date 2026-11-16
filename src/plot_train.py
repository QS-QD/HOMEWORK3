# -*- coding: utf-8 -*-
"""plot_train.py — 训练过程可视化
产出: assets/fig4_loss.png  fig5_lr_grad.png  fig6_samples.png  fig7_params.png
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

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
GRID = dict(color="#e8eaee", linewidth=0.8, zorder=0)


def load(tag="lstm"):
    with open(os.path.join(OUT, f"{tag}_metrics.json"), encoding="utf-8") as f:
        return json.load(f)


def smooth(y, k=25):
    if len(y) < k:
        return np.array(y)
    return np.convolve(y, np.ones(k) / k, mode="valid")


def fig_loss(m, tag=""):
    h = m["history"]
    ep = [x["epoch"] for x in h]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # (a) 损失曲线（按轮次）
    ax = axes[0]
    tr = [x["train_loss"] for x in h]
    va = [x["val_loss"] for x in h]
    ax.plot(ep, tr, "o-", color=C1, ms=3.5, lw=1.8, label="训练损失")
    ax.plot(ep, va, "s-", color=C2, ms=3.2, lw=1.8, label="验证损失")
    ax.fill_between(ep, tr, va, color="#9bb4c9", alpha=0.16, zorder=0,
                    label="泛化间隙")
    best = min(h, key=lambda x: x["val_loss"])
    ax.scatter([best["epoch"]], [best["val_loss"]], s=110, facecolor="none",
               edgecolor=C3, lw=2, zorder=5,
               label=f"最佳验证（第 {best['epoch']} 轮）")
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("交叉熵损失")
    ax.set_title("(a) 损失曲线：两者始终贴合，几乎不过拟合", fontsize=11, pad=8)
    ax.grid(**GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_xlim(0, max(ep) + 1)

    # (b) 困惑度与准确率
    ax = axes[1]
    ax.plot(ep, [x["val_ppl"] for x in h], "o-", color=C2, ms=3.5, lw=2,
            label="验证困惑度 PPL")
    ax.set_yscale("log")
    ppls = [x["val_ppl"] for x in h]
    ax.set_ylim(min(ppls) * 0.82, max(ppls) * 1.18)
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("困惑度（对数轴）", color=C2)
    ax.tick_params(axis="y", labelcolor=C2)
    ax.grid(**GRID)
    ax.set_title("(b) 困惑度与下一字预测准确率", fontsize=11, pad=8)

    ax2 = ax.twinx()
    ax2.plot(ep, [x["val_acc"] * 100 for x in h], "s-", color=C3, ms=3.5, lw=1.8,
             label="Top-1 准确率")
    ax2.set_ylabel("下一字预测准确率 (%)", color=C3)
    ax2.tick_params(axis="y", labelcolor=C3)
    ax2.set_ylim(0, max(x["val_acc"] for x in h) * 100 * 1.25)

    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, frameon=False, fontsize=8.5, loc="center right")

    plt.tight_layout()
    p = os.path.join(ASSETS, f"fig4_loss{tag}.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)


def fig_lr_grad(m, tag=""):
    st = m["steps"]
    steps = [s["step"] for s in st]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

    ax = axes[0]
    ax.plot(steps, [s["lr"] for s in st], color=C1, lw=1.8)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("学习率")
    ax.set_title("(a) OneCycle 学习率调度：先热身再余弦衰减",
                 fontsize=11, pad=8)
    ax.grid(**GRID)
    peak = max(range(len(steps)), key=lambda i: st[i]["lr"])
    ax.annotate(f"峰值 {st[peak]['lr']:.5f}", xy=(steps[peak], st[peak]["lr"]),
                xytext=(steps[peak] + len(steps) * 0.08, st[peak]["lr"] * 0.92),
                fontsize=9, color=C1,
                arrowprops=dict(arrowstyle="->", color=C1, lw=1))

    ax = axes[1]
    g = [s["grad_norm"] for s in st]
    ax.plot(steps, g, color=C2, alpha=0.25, lw=0.8, label="梯度范数")
    sm = smooth(g, 25)
    ax.plot(range(25, 25 + len(sm)), sm, color=C2, lw=2, label="平滑后")
    ax.axhline(1.0, color=C3, ls="--", lw=1.2, label="裁剪阈值 1.0")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("梯度范数")
    ax.set_title("(b) 梯度范数：始终被裁剪阈值约束，训练稳定",
                 fontsize=11, pad=8)
    ax.grid(**GRID)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    p = os.path.join(ASSETS, f"fig5_lr_grad{tag}.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)


def fig_samples(m, tag=""):
    """生成文本随训练轮次的演进"""
    samples = m["samples"]
    by_ep = {s["epoch"]: s for s in samples}
    # 关键节点：乱码 → 突然成型 → 逐渐雅化
    want = [1, 3, 5, 7, 14, 27, 40]
    picks = [by_ep[e] for e in want if e in by_ep]
    if len(picks) < 5:                      # 兜底：轮次不足时均匀取
        idx = np.linspace(0, len(samples) - 1, min(7, len(samples))).round().astype(int)
        picks = [samples[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(picks) * 1.0 + 0.4)
    ax.axis("off")

    ax.text(0.15, len(picks) * 1.0 + 0.15, "同一个开头，训练不同阶段的生成结果",
            fontsize=13, weight="bold", color="#26313d")

    for i, s in enumerate(picks[::-1]):
        y = 0.25 + i * 1.0
        ax.add_patch(FancyBboxPatch((0.15, y), 9.7, 0.82,
                                    boxstyle="round,pad=0.01,rounding_size=0.08",
                                    facecolor="#f7f9fb" if i % 2 else "#ffffff",
                                    edgecolor="#e2e6ec", lw=0.8, zorder=1))
        ax.text(0.42, y + 0.55, f"{s['epoch']:>2d}", fontsize=12, weight="bold",
                color=C1, ha="center", va="center", zorder=2)
        ax.text(0.42, y + 0.26, "轮", fontsize=7.5, color="#9aa3b0",
                ha="center", va="center", zorder=2)
        txt = s["text"].replace("\n", "｜").replace("｜｜", "｜")
        show = txt[:46] + ("…" if len(txt) > 46 else "")
        ax.text(0.85, y + 0.42, show, fontsize=9.8, color="#26313d",
                va="center", zorder=2, family=plt.rcParams["font.sans-serif"][0])

    plt.tight_layout()
    p = os.path.join(ASSETS, f"fig6_samples{tag}.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)


def fig_params(m, tag=""):
    pb = m["params"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    ax = axes[0]
    labels = ["Embedding\n0.80 M", "LSTM\n0.39 M", "Linear(输出层)\n1.59 M"]
    vals = [pb["embed"], pb["lstm"], pb["fc"]]
    cols = [C1, C2, C3]
    w, _, at = ax.pie(vals, labels=labels, colors=cols, autopct="%1.1f%%",
                      startangle=90, textprops=dict(fontsize=9.5),
                      wedgeprops=dict(width=0.45, edgecolor="white", lw=1.5),
                      pctdistance=0.78)
    for t in at:
        t.set_fontsize(8.5)
        t.set_color("white")
    ax.text(0, 0, f"{pb['total'] / 1e6:.2f} M\n总参数", ha="center", va="center",
            fontsize=11, weight="bold", color="#26313d")
    ax.set_title("(a) 参数构成：输出层占比最大", fontsize=11, pad=8)

    ax = axes[1]
    ax.axis("off")
    cfg = m["config"]
    rows = [("模型结构", f"Embedding → LSTM×{cfg['num_layers']} → Linear"),
            ("嵌入/隐藏维度", f"{cfg['embed_dim']} / {cfg['hidden_dim']}"),
            ("序列长度", f"{cfg['seq_len']} 字"),
            ("批大小", str(cfg["batch_size"])),
            ("优化器", "AdamW (weight_decay=0.01)"),
            ("学习率", f"OneCycle, 峰值 {cfg['lr']}"),
            ("训练轮次", f"{cfg['epochs']} 轮"),
            ("训练设备", m["device"]),
            ("训练总耗时", f"{m['total_time']:.0f} 秒")]
    ax.text(0.02, 0.96, "训练配置", fontsize=12, weight="bold", color="#26313d")
    for i, (k, v) in enumerate(rows):
        y = 0.86 - i * 0.098
        ax.text(0.04, y, k, fontsize=9.2, color="#5b6472", va="center")
        ax.text(0.36, y, v, fontsize=9.2, color="#26313d", va="center",
                weight="bold")
        ax.plot([0.02, 0.95], [y - 0.038, y - 0.038], color="#eef0f3", lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    p = os.path.join(ASSETS, f"fig7_params{tag}.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "lstm"
    m = load(tag)
    fig_loss(m, "" if tag == "lstm" else "_" + tag)
    fig_lr_grad(m, "" if tag == "lstm" else "_" + tag)
    fig_samples(m, "" if tag == "lstm" else "_" + tag)
    fig_params(m, "" if tag == "lstm" else "_" + tag)
