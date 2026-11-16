# -*- coding: utf-8 -*-
"""
generate.py — 用训练好的模型写诗

用法:
    python src/generate.py --prompt 春風 --n 60 --temperature 0.8
    python src/generate.py --prompt 山中 --temperature 0.6 --top-k 5
"""
import argparse
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import CharLSTM  # noqa: E402


def load_model(tag="lstm", device="cpu"):
    path = os.path.join(ROOT, "outputs", f"{tag}_best.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = CharLSTM(ck["vocab_size"], cfg["embed_dim"], cfg["hidden_dim"],
                     cfg["num_layers"], cfg["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    vocab = ck["vocab"]
    itos = {i: c for c, i in vocab.items()}
    return model, vocab, itos, ck


def format_poem(text):
    """把连续生成的字整理成诗行"""
    text = text.replace("<e>", "\n").replace("\n", "")
    # 按句号/问号切句
    parts = re.split(r"(?<=[。？！])", text)
    lines, cur = [], ""
    for p in parts:
        cur += p
        if len(cur) >= 12:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines


def write(model, vocab, itos, prompt, n, temperature, top_k, device):
    idx = torch.tensor([[vocab[c] for c in prompt if c in vocab]],
                       dtype=torch.long, device=device)
    ids = model.generate(idx, n, temperature=temperature, top_k=top_k)
    body = "".join(itos.get(i, "") for i in ids)
    return prompt + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="春風")
    ap.add_argument("--n", type=int, default=56)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--tag", default="lstm")
    ap.add_argument("--num", type=int, default=1, help="生成几首")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, itos, ck = load_model(args.tag, device)
    print(f"模型: {args.tag} (第 {ck['epoch']} 轮, 验证损失 {ck['val_loss']:.4f})\n")

    for i in range(args.num):
        txt = write(model, vocab, itos, args.prompt, args.n,
                    args.temperature, args.top_k, device)
        print(f"—— 第 {i + 1} 首 " + "—" * 30)
        for ln in format_poem(txt):
            print("  " + ln)
        print()


if __name__ == "__main__":
    main()
