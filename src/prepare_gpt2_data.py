# -*- coding: utf-8 -*-
"""
prepare_gpt2_data.py — 为 GPT-2 微调准备语料

和从零训练用的 poetry.txt 不同, 这里的分词器是 uer/gpt2-chinese-poem 自带的
BertTokenizer (字符级, 21128 词表), 所以要做两件事:

1. 每首诗压成一行: 去掉句间换行, 保留中文标点
2. 诗与诗之间插入 [SEP] 作为边界, 让模型知道"一首诗到这里结束"

划分 train/val 时以「首」为单位切分, 而不是按 token 块切,
否则同一首诗的前半段在训练集、后半段在验证集会高估效果。

输出: data/gpt2/train.txt  data/gpt2/val.txt  data/gpt2/meta.json
"""
import json
import os
import sys

from zhconv import convert

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import load_poems

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "gpt2")
SEP = "[SEP]"

# 训练集 / 验证集按诗切分
VAL_RATIO = 0.08
SEED = 42


def flatten(lines):
    """把一首诗的若干行压成连续文本, 句间直接由标点隔开"""
    return "".join(lines)


def to_simplified(s):
    """繁 -> 简

    uer/gpt2-chinese-poem 的词表虽然收录了繁体字, 但预训练语料以简体为主,
    繁体字对应的 embedding 几乎没被更新过。实测同一句诗, 繁体输入的困惑度
    比简体高一个数量级, 所以这里统一转成简体, 让微调真正学到"风格"而不是
    "补认字"。
    """
    return convert(s, "zh-hans")


def main():
    poems = load_poems()
    print(f"[1/4] 读到诗 {len(poems)} 首")

    # 按固定种子打乱后切分, 保证可复现
    import random

    rng = random.Random(SEED)
    idx = list(range(len(poems)))
    rng.shuffle(idx)
    n_val = int(len(poems) * VAL_RATIO)
    val_idx, train_idx = set(idx[:n_val]), set(idx[n_val:])

    train_texts, val_texts = [], []
    n_converted = 0
    for i, (title, author, lines) in enumerate(poems):
        body = flatten(lines)
        if not body:
            continue
        simp = to_simplified(body)
        if simp != body:
            n_converted += 1
        (val_texts if i in val_idx else train_texts).append(simp)

    # 拼成流: 每首诗以 [SEP] 收尾
    train_flow = "".join(t + SEP for t in train_texts)
    val_flow = "".join(t + SEP for t in val_texts)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "train.txt"), "w", encoding="utf-8") as f:
        f.write(train_flow)
    with open(os.path.join(OUT_DIR, "val.txt"), "w", encoding="utf-8") as f:
        f.write(val_flow)

    meta = {
        "num_poems_total": len(poems),
        "num_poems_train": len(train_texts),
        "num_poems_val": len(val_texts),
        "chars_train": len(train_flow),
        "chars_val": len(val_flow),
        "num_poems_converted": n_converted,
        "sep": SEP,
        "val_ratio": VAL_RATIO,
        "seed": SEED,
        "avg_poem_chars": round(sum(len(t) for t in train_texts) / len(train_texts), 1),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[2/4] 训练 {len(train_texts)} 首 / 验证 {len(val_texts)} 首")
    print(f"[3/4] 字符数: 训练 {len(train_flow):,} / 验证 {len(val_flow):,}")
    print(f"[4/4] 平均每首 {meta['avg_poem_chars']} 字 | 繁->简改写 {n_converted} 首")
    print("\n预览:", train_flow[:160])


if __name__ == "__main__":
    main()
