# -*- coding: utf-8 -*-
"""plot_eval.py — 模型分析可视化
产出: assets/fig8_embed.png  fig9_nextchar.png  fig10_temperature.png  fig11_topk.png
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import CharLSTM  # noqa: E402
from generate import load_model, format_poem  # noqa: E402

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


def fig_embed(model, itos, stats):
    """字符嵌入的 PCA 二维投影"""
    W = model.embed.weight.detach().cpu().numpy()
    freq = [(c, n) for c, n in stats["top_chars"]
            if "\u4e00" <= c <= "\u9fff"][:350]

    # PCA
    X = W - W.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.scatter(Z[:, 0], Z[:, 1], s=6, color="#c8d3e0", alpha=0.55, zorder=1)

    # 标注若干高频字
    show = [c for c, _ in freq[:22]]
    for c in show:
        i = model_vocab[c]
        ax.scatter(Z[i, 0], Z[i, 1], s=30, color=C2, zorder=2)
        ax.text(Z[i, 0], Z[i, 1] + 0.03, c, fontsize=10, color="#26313d",
                ha="center", va="bottom", zorder=3)
    ax.set_xlabel(f"主成分 1（解释方差 {S[0] ** 2 / (S ** 2).sum() * 100:.1f}%）")
    ax.set_ylabel(f"主成分 2（解释方差 {S[1] ** 2 / (S ** 2).sum() * 100:.1f}%）")
    ax.set_title("字符嵌入的前两个主成分：二维投影只保留少量信息，语义聚类并不明显",
                 fontsize=11, pad=10)
    ax.grid(**GRID)
    plt.tight_layout()
    p = os.path.join(ASSETS, "fig8_embed.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)


@torch.no_grad()
def fig_nextchar(model, vocab, itos):
    """给定上文，看模型预测的下一个字（只显示汉字候选）"""
    cases = ["舉頭望明", "春風吹又", "白日依山", "故人西辭"]
    fig, axes = plt.subplots(1, len(cases), figsize=(13, 3.7))

    # 非汉字位置屏蔽
    cjk_mask = torch.zeros(len(itos), dtype=torch.bool)
    for i, c in itos.items():
        if "\u4e00" <= c <= "\u9fff":
            cjk_mask[i] = True

    for ax, ctx in zip(axes, cases):
        idx = torch.tensor([[vocab[c] for c in ctx if c in vocab]], dtype=torch.long)
        logits, _ = model(idx)
        prob = torch.softmax(logits[0, -1], -1) * cjk_mask
        top = torch.topk(prob, 6)
        chars = [itos[i] for i in top.indices.tolist()]
        vals = [v * 100 for v in top.values.tolist()]

        colors = [C2] + [C1] * 5
        bars = ax.bar(range(len(chars)), vals, color=colors, width=0.62, zorder=3)
        ax.set_xticks(range(len(chars)))
        ax.set_xticklabels(chars, fontsize=14)
        ax.set_ylim(0, max(vals) * 1.32)
        ax.set_title(f"「{ctx}」→", fontsize=12, pad=6)
        ax.set_ylabel("概率 (%)" if ax is axes[0] else "")
        ax.grid(axis="y", **GRID)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03,
                    f"{v:.1f}", ha="center", fontsize=8.5, color="#5b6472")

    plt.suptitle("模型学到了什么：给定上文，下一个字（仅汉字候选）的概率分布",
                 fontsize=13, weight="bold", y=1.0)
    plt.tight_layout()
    p = os.path.join(ASSETS, "fig9_nextchar.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)


def fig_temperature(model, vocab, itos):
    prompt = "春風"
    temps = [0.5, 0.8, 1.0, 1.3]
    outs = []
    for t in temps:
        idx = torch.tensor([[vocab[c] for c in prompt]], dtype=torch.long)
        ids = model.generate(idx, 44, temperature=t)
        outs.append(prompt + "".join(itos.get(i, "") for i in ids))

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(temps) * 1.15 + 0.3)
    ax.axis("off")
    ax.text(0.1, len(temps) * 1.15 + 0.05,
            "温度参数对生成结果的影响（同一个开头「春風」）",
            fontsize=13, weight="bold", color="#26313d")

    desc = {0.5: "保守：反复用最常见的字，句子通顺但保守呆板",
            0.8: "平衡：结构完整，偶有新意（默认）",
            1.0: "接近原始分布：更像语料，但偶有生硬搭配",
            1.3: "发散：字词跳跃，易出现不连贯的句子"}
    for i, (t, txt) in enumerate(zip(temps, outs)):
        y = 0.2 + (len(temps) - 1 - i) * 1.15
        ax.add_patch(FancyBboxPatch((0.1, y), 9.75, 0.95,
                                    boxstyle="round,pad=0.01,rounding_size=0.08",
                                    facecolor="#f7f9fb" if i % 2 == 0 else "#ffffff",
                                    edgecolor="#e2e6ec", lw=0.8))
        ax.text(0.35, y + 0.62, f"T={t}", fontsize=11.5, weight="bold", color=C1,
                ha="center", va="center")
        ax.text(0.35, y + 0.3, desc[t][:2], fontsize=7.5, color="#9aa3b0",
                ha="center", va="center")
        body = txt.replace("<e>", "").replace("\n", "")
        show = body[:42] + ("…" if len(body) > 42 else "")
        ax.text(0.85, y + 0.72, show, fontsize=9.8, color="#26313d", va="center")
        ax.text(0.85, y + 0.36, desc[t], fontsize=8.5, color="#7b8794", va="center")

    plt.tight_layout()
    p = os.path.join(ASSETS, "fig10_temperature.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)


@torch.no_grad()
def fig_topk(model, val_arr, vocab_size, device):
    """Top-k 预测准确率 + 预测熵分布"""
    model.eval()
    ks = [1, 3, 5, 10, 20, 50]
    hit = {k: 0 for k in ks}
    total = 0
    ents = []
    for i in range(0, min(len(val_arr), 200), 8):
        b = torch.from_numpy(val_arr[i:i + 8].astype(np.int64))
        x, y = b[:, :-1], b[:, 1:]
        logits, _ = model(x)
        logits = logits.reshape(-1, logits.size(-1))
        y = y.reshape(-1)
        prob = torch.softmax(logits, -1)
        ents.append((-torch.log(prob.gather(1, y[:, None]) + 1e-9)).mean().item())
        for k in ks:
            top = logits.topk(k, dim=-1).indices
            hit[k] += (top == y[:, None]).any(-1).sum().item()
        total += y.numel()

    accs = [hit[k] / total * 100 for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

    ax = axes[0]
    bars = ax.bar([str(k) for k in ks], accs, color=C1, width=0.6, zorder=3)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.2, f"{a:.1f}%",
                ha="center", fontsize=9.5, color="#26313d")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("命中准确率 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Top-k 命中率：前 1 名约 1/3，前 50 名过半",
                 fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)

    ax = axes[1]
    ax.hist(ents, bins=18, color=C2, zorder=3, edgecolor="white")
    ax.axvline(np.mean(ents), color=C3, ls="--", lw=1.6,
               label=f"平均值 {np.mean(ents):.2f} nat")
    ax.set_xlabel("平均交叉熵损失（nat/字）")
    ax.set_ylabel("验证批次数")
    ax.set_title("(b) 各批次的预测误差分布", fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    p = os.path.join(ASSETS, "fig11_topk.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)
    return accs


model_vocab = {}


def main():
    global model_vocab
    tag = sys.argv[1] if len(sys.argv) > 1 else "lstm"
    device = "cpu"
    model, vocab, itos, ck = load_model(tag, device)
    model_vocab = vocab

    with open(os.path.join(ROOT, "data", "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)

    fig_embed(model, itos, stats)
    fig_nextchar(model, vocab, itos)
    fig_temperature(model, vocab, itos)

    # Top-k 分析需要验证集
    from train import load_data
    _, val_arr, _, _ = load_data(128)
    accs = fig_topk(model, val_arr, len(vocab), device)
    print("Top-k 准确率:", [f"{k}:{a:.1f}%" for k, a in
                            zip([1, 3, 5, 10, 20, 50], accs)])


if __name__ == "__main__":
    main()
