# -*- coding: utf-8 -*-
"""
eval_ood.py — 灾难性遗忘检测

微调会把模型的参数往"唐诗"这个方向拽。问题是拽得太狠时, 它原本掌握的
一般中文能力会被覆盖掉, 这叫灾难性遗忘(catastrophic forgetting)。

这里用一份与唐诗完全无关的现代汉语说明文(约 800 字, data/ood_modern_zh.txt)
作为域外测试集, 分别测微调前 / 微调后的困惑度。

产出: outputs/ood_metrics.json
"""
import json
import math
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "gpt2-chinese-poem")
OUT = os.path.join(ROOT, "outputs")
OOD = os.path.join(ROOT, "data", "ood_modern_zh.txt")


def load(ckpt, device):
    from transformers import BertTokenizer, GPT2LMHeadModel

    tok = BertTokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    if ckpt != "base":
        model.load_state_dict(torch.load(os.path.join(OUT, f"{ckpt}.pt"), map_location="cpu"))
    return model.to(device).eval(), tok


@torch.no_grad()
def ppl_of(model, tok, text, device):
    ids = tok.encode(text, add_special_tokens=False)[:1023]
    x = torch.tensor([ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(input_ids=x[:, :-1], labels=x[:, :-1])
    loss = out.loss.item()
    return loss, math.exp(min(loss, 20)), len(ids)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(OOD, encoding="utf-8") as f:
        paras = [p.strip() for p in f.read().split("\n") if p.strip()]
    print(f"域外文本: {len(paras)} 段, {sum(len(p) for p in paras)} 字")

    result = {"paragraphs": [], "models": {}}
    for tag in ["base", "ft_best"]:
        model, tok = load(tag, device)
        rows = []
        for p in paras:
            loss, ppl, n = ppl_of(model, tok, p, device)
            rows.append({"len": n, "loss": round(loss, 3), "ppl": round(ppl, 1)})
        avg_ppl = sum(math.log(r["ppl"]) for r in rows) / len(rows)
        result["models"][tag] = {
            "rows": rows,
            "mean_ppl": round(math.exp(avg_ppl), 1),
        }
        print(f"{tag:8s} 平均困惑度 {math.exp(avg_ppl):8.1f}  " +
              " ".join(f"{r['ppl']:.0f}" for r in rows))
        del model
        torch.cuda.empty_cache()

    result["paragraphs"] = [p[:40] + "…" for p in paras]
    with open(os.path.join(OUT, "ood_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    b = result["models"]["base"]["mean_ppl"]
    f_ = result["models"]["ft_best"]["mean_ppl"]
    print(f"\n变化: {b:.1f} -> {f_:.1f}  ({'+' if f_ > b else ''}{(f_ - b) / b * 100:.1f}%)")


if __name__ == "__main__":
    main()
