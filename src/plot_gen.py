# -*- coding: utf-8 -*-
"""plot_gen.py — 生成效果可视化
产出: assets/g3_compare.png  微调前后同一开头的续写对比
      assets/g4_temp.png     不同温度下的生成结果
"""
import json
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models", "gpt2-chinese-poem")

for _f in ["C:/Windows/Fonts/msyh.ttc"]:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=_f).get_name()]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

C1, C2, C3 = "#2f6f9f", "#c0642f", "#3f8f6f"
GREY = "#8b95a3"

SEEDS = ["春风", "孤舟", "明月", "故人"]


def load(ckpt, device="cuda"):
    from transformers import BertTokenizer, GPT2LMHeadModel
    tok = BertTokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    if ckpt != "base":
        model.load_state_dict(torch.load(os.path.join(OUT, f"{ckpt}.pt"), map_location="cpu"))
    return model.to(device).eval(), tok


@torch.no_grad()
def gen(model, tok, prompt, n, temp, device="cuda", seed=0):
    torch.manual_seed(seed)
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    # min_new_tokens: 模型有时在开头两个字之后就吐 [SEP], 强制它写够一整首
    out = model.generate(x, max_new_tokens=n, min_new_tokens=44, do_sample=True,
                         temperature=temp, top_p=0.9, top_k=0,
                         repetition_penalty=1.08,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.sep_token_id)
    s = "".join(tok.convert_ids_to_tokens(i) for i in out[0][len(ids):].tolist())
    return s.replace("[SEP]", "\n").replace("[PAD]", "").replace("[UNK]", "□")


def gen_clean(model, tok, prompt, n, temp, device="cuda", seed=0):
    """生成并跳过"标点开头"的样本

    强制 min_new_tokens 后, 模型偶尔会用逗号、句号接茬("孤舟，于此意何深"),
    语义上没错, 但展示效果差, 换个随机种子重采。
    """
    s = ""
    for k in range(6):
        s = gen(model, tok, prompt, n, temp, device, seed + k)
        if s and s[0] not in "，。！？、；：":
            break
    return s


def lines_of(s, per=11):
    """按句号断行, 每行尽量不超过 per*2 字"""
    out, cur = [], ""
    for ch in s:
        cur += ch
        if ch in "。！？" and len(cur) >= per:
            out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur)
    return out[:3]


def fig_compare():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base, tok = load("base", dev)
    ft, _ = load("ft_best", dev)

    fig, axes = plt.subplots(len(SEEDS), 2, figsize=(12.6, 8.6))
    for i, s in enumerate(SEEDS):
        for j, (model, name, col) in enumerate(
                [(base, "微调前（原始 gpt2-chinese-poem）", GREY),
                 (ft, "微调后", C1)]):
            ax = axes[i][j]
            ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor="#f7f9fb" if j else "white",
                                       edgecolor="#e2e6ec", lw=1.0))
            txt = s + gen_clean(model, tok, s, 60, 0.85, dev, seed=i)
            body = lines_of(txt, 12)
            ax.text(0.035, 0.78, s, fontsize=13, weight="bold", color=col,
                    transform=ax.transAxes, va="top")
            ax.text(0.035, 0.60, "\n".join(body), fontsize=10.4, color="#26313d",
                    transform=ax.transAxes, va="top", linespacing=1.65)
            if i == 0:
                ax.set_title(name, fontsize=11.5, color=col, pad=9)

    fig.suptitle("同一个开头，微调前后的续写差异（温度 0.85）",
                 fontsize=13.5, weight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(ASSETS, "g3_compare.png")
    plt.savefig(p, dpi=165, bbox_inches="tight")
    plt.close()
    print("saved", p)
    del base, ft
    torch.cuda.empty_cache()


TEMPS = [(0.5, "T = 0.5　保守，几乎复述常见句式"),
         (0.85, "T = 0.85　平衡（默认）"),
         (1.2, "T = 1.2　放开，用字变生僻、句意发散")]


def fig_temp():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load("ft_best", dev)

    fig, axes = plt.subplots(len(TEMPS), 1, figsize=(11.8, 6.4))
    for ax, (t, label) in zip(axes, TEMPS):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor="#f7f9fb", edgecolor="#e2e6ec", lw=1.0))
        txt = "春风" + gen_clean(model, tok, "春风", 56, t, dev, seed=7)
        body = "".join(lines_of(txt, 13)[:2])
        ax.text(0.028, 0.72, label, fontsize=10.6, weight="bold", color=C2,
                transform=ax.transAxes, va="top")
        ax.text(0.028, 0.40, body, fontsize=10.6, color="#26313d",
                transform=ax.transAxes, va="top")

    fig.suptitle("采样温度对生成结果的影响（微调后模型，固定开头「春风」）",
                 fontsize=13, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(ASSETS, "g4_temp.png")
    plt.savefig(p, dpi=165, bbox_inches="tight")
    plt.close()
    print("saved", p)


if __name__ == "__main__":
    fig_compare()
    fig_temp()
