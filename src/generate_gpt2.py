# -*- coding: utf-8 -*-
"""
generate_gpt2.py — 用微调后的模型写诗

用法:
    python src/generate_gpt2.py --prompt 春风 --n 60
    python src/generate_gpt2.py --prompt 明月 --temperature 0.7 --ckpt ft_best
    python src/generate_gpt2.py --prompt 孤舟 --ckpt base      # 用未微调的原模型
"""
import argparse
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "gpt2-chinese-poem")
OUT = os.path.join(ROOT, "outputs")


def load(ckpt="ft_best", device="cuda"):
    """ckpt: 'base' 表示未微调的原始权重, 否则读取 outputs/<ckpt>.pt"""
    from transformers import BertTokenizer, GPT2LMHeadModel

    tok = BertTokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    if ckpt != "base":
        path = os.path.join(OUT, f"{ckpt}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到权重 {path}")
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    return model.to(device).eval(), tok


def ids_to_text(tok, ids):
    s = "".join(tok.convert_ids_to_tokens(i) for i in ids)
    return (s.replace("[SEP]", "\n").replace("[PAD]", "")
             .replace("[UNK]", "□").replace("[CLS]", ""))


@torch.no_grad()
def write(model, tok, prompt, n=60, temperature=0.9, top_p=0.9, device="cuda"):
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x, max_new_tokens=n,
        # 不设下限时模型常写两个字就吐 [SEP], 得到的残句没有参考价值
        min_new_tokens=max(20, n // 2),
        do_sample=True, temperature=temperature,
        top_p=top_p, top_k=0, repetition_penalty=1.08,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.sep_token_id,
    )
    return ids_to_text(tok, out[0][len(ids):].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="春风")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--ckpt", type=str, default="ft_best")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.ckpt, device)
    print(f"权重: {args.ckpt} | 开头「{args.prompt}」| 温度 {args.temperature}\n")
    for i in range(args.num):
        txt = write(model, tok, args.prompt, args.n, args.temperature, args.top_p, device)
        print(f"--- 样本 {i+1} ---")
        print(f"{args.prompt}{txt}\n")


if __name__ == "__main__":
    main()
