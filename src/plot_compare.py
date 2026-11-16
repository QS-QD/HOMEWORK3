# -*- coding: utf-8 -*-
"""plot_compare.py — 模型规模对比 -> assets/fig12_compare.png"""
import json
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from generate import load_model  # noqa: E402

ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "outputs")

for _f in ["C:/Windows/Fonts/msyh.ttc"]:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=_f).get_name()]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

C1, C2, C3 = "#2f6f9f", "#c0642f", "#3f8f6f"
GRID = dict(color="#e8eaee", linewidth=0.8, zorder=0)

TAGS = [("small", "小"), ("lstm", "中"), ("big", "大")]


def collect():
    rows = []
    for tag, label in TAGS:
        m = json.load(open(os.path.join(OUT, f"{tag}_metrics.json"), encoding="utf-8"))
        best = min(m["history"], key=lambda x: x["val_loss"])
        cfg = m["config"]
        rows.append({
            "tag": tag, "label": label,
            "params": m["params"]["total"] / 1e6,
            "ppl": best["val_ppl"], "acc": best["val_acc"] * 100,
            "loss": best["val_loss"], "time": m["total_time"],
            "embed": cfg["embed_dim"], "hidden": cfg["hidden_dim"],
        })
    return rows


def gen_samples(n=40):
    """用三个模型在同样条件下各生成一段"""
    outs = []
    for tag, label in TAGS:
        torch.manual_seed(7)
        model, vocab, itos, ck = load_model(tag, "cpu")
        idx = torch.tensor([[vocab[c] for c in "春風"]], dtype=torch.long)
        ids = model.generate(idx, n, temperature=0.8)
        body = "".join(itos.get(i, "") for i in ids).replace("<e>", "").replace("\n", "")
        while body and body[0] in "，。！？；：、":
            body = body[1:]
        outs.append(body)
    return outs


def main():
    rows = collect()
    samples = gen_samples()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    # (a) 参数量 vs 困惑度
    ax = axes[0]
    xs = [r["params"] for r in rows]
    ys = [r["ppl"] for r in rows]
    ax.plot(xs, ys, "o-", color=C1, ms=9, lw=2, zorder=3)
    for r in rows:
        ax.annotate(f'{r["ppl"]:.1f}', (r["params"], r["ppl"]),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=11, weight="bold", color=C1)
        ax.annotate(r["label"], (r["params"], r["ppl"]),
                    textcoords="offset points", xytext=(0, -22),
                    ha="center", fontsize=9.5, color="#7b8794")
    ax.set_xlabel("参数量（百万）")
    ax.set_ylabel("验证困惑度")
    ax.set_title("(a) 参数量翻 3.4 倍，困惑度只降 30%", fontsize=11, pad=8)
    ax.grid(**GRID)
    ax.set_ylim(min(ys) - 14, max(ys) + 12)
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.6)

    # (b) 生成质量对比
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.55)
    ax.axis("off")

    for i, (r, s) in enumerate(zip(rows, samples)):
        y = 2.55 - i * 1.05
        ax.add_patch(FancyBboxPatch((0.05, y), 9.85, 0.88,
                                    boxstyle="round,pad=0.01,rounding_size=0.08",
                                    facecolor="#f7f9fb" if i % 2 == 0 else "#ffffff",
                                    edgecolor="#e2e6ec", lw=0.8))
        ax.text(0.42, y + 0.60, r["label"], fontsize=11, weight="bold",
                color=C1, ha="center", va="center")
        ax.text(0.42, y + 0.28, f'{r["params"]:.2f}M', fontsize=7.8,
                color="#9aa3b0", ha="center", va="center")
        ax.text(0.85, y + 0.58, s[:38], fontsize=9.2, color="#26313d", va="center")
        ax.text(0.85, y + 0.24, f'困惑度 {r["ppl"]:.1f} · 准确率 {r["acc"]:.1f}%',
                fontsize=8, color="#7b8794", va="center")

    ax.text(0.05, 3.35, "同一开头「春風」（温度 0.8）三个规模的续写", fontsize=11,
            weight="bold", color="#26313d")

    plt.tight_layout()
    p = os.path.join(ASSETS, "fig12_compare.png")
    plt.savefig(p, dpi=160)
    plt.close()
    print("saved", p)
    for r, s in zip(rows, samples):
        print(f'{r["label"]:2s} {r["params"]:.2f}M ppl {r["ppl"]:.1f} → {s[:50]}')


if __name__ == "__main__":
    main()
