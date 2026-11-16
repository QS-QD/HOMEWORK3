# -*- coding: utf-8 -*-
"""
prepare_data.py — 构建字符级语料与词表

输入: data/raw/poet.tang.*.json   (chinese-poetry《全唐诗》分卷)
输出: data/poetry.txt             训练语料(每行一句诗)
      data/vocab.json            字符<->编号映射
      data/stats.json            语料统计(供报告画图)
"""
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
DATA_DIR = os.path.join(ROOT, "data")

# 只保留中文字符与常见中文标点
CJK = r"[\u4e00-\u9fff]"
PUNCT = "，。！？；：、"


def load_poems():
    """读取所有分卷, 返回 (标题, 作者, [诗句...]) 列表"""
    poems = []
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".json"))
    for fn in files:
        with open(os.path.join(RAW_DIR, fn), encoding="utf-8") as f:
            for item in json.load(f):
                paras = item.get("paragraphs") or []
                lines = []
                for p in paras:
                    p = p.strip()
                    if not p:
                        continue
                    # 丢弃夹杂乱码/非中文的行
                    if not re.search(CJK, p):
                        continue
                    # 只允许中文与中文标点
                    if re.search(r"[^\u4e00-\u9fff" + PUNCT + "]", p):
                        continue
                    lines.append(p)
                if lines:
                    poems.append((item.get("title", ""), item.get("author", ""), lines))
    return poems


def build():
    poems = load_poems()
    print(f"[1/4] 读到诗 {len(poems)} 首")

    # ---- 语料拼接: 每句一行, 诗之间空行 ----
    chunks = []
    for _, _, lines in poems:
        chunks.append("\n".join(lines))
    text = "\n\n".join(chunks)

    # ---- 字符表 ----
    # 一行 = 一首诗, 用特殊符号标记诗的开始/结束, 让模型学会"起承转合"
    # 这里使用 <e> 表示一首诗结束
    text = "\n\n".join(c + "\n<e>" for c in chunks)

    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in vocab.items()}

    # ---- 统计 ----
    line_counter = Counter()
    char_counter = Counter()
    poem_len = []
    for _, _, lines in poems:
        poem_len.append(len(lines))
        for ln in lines:
            line_counter[len(ln)] += 1
            char_counter.update(ln)

    stats = {
        "num_poems": len(poems),
        "num_lines": sum(poem_len),
        "num_chars": len(text),
        "vocab_size": len(vocab),
        "top_chars": char_counter.most_common(40),
        "line_len_dist": dict(sorted(line_counter.items())),
        "poem_len_dist": dict(sorted(Counter(poem_len).items())),
        "char_freq_rank": [c for c, _ in char_counter.most_common(600)],
        "char_freq_count": [n for _, n in char_counter.most_common(600)],
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "poetry.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(DATA_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "itos": {str(k): v for k, v in itos.items()}},
                  f, ensure_ascii=False)
    with open(os.path.join(DATA_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)

    print(f"[2/4] 诗 {stats['num_poems']} 首 / 句 {stats['num_lines']} 行")
    print(f"[3/4] 总字符 {stats['num_chars']:,} / 词表 {stats['vocab_size']}")
    print(f"[4/4] 高频字: {''.join(c for c, _ in stats['top_chars'][:30])}")
    print("\n语料预览:")
    print(text[:200])


if __name__ == "__main__":
    build()
