# -*- coding: utf-8 -*-
"""
download_model.py — 从 Hugging Face 镜像下载 gpt2-chinese-poem

huggingface.co 在国内经常连不上, 所以默认走 hf-mirror.com。
下载 5 个文件: config / vocab / tokenizer 配置 / 权重(约 405 MB)。

用法:
    python src/download_model.py
"""
import os
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "models", "gpt2-chinese-poem")
REPO = "uer/gpt2-chinese-poem"
MIRRORS = ["https://hf-mirror.com", "https://huggingface.co"]
FILES = ["config.json", "vocab.txt", "tokenizer_config.json",
         "special_tokens_map.json", "pytorch_model.bin"]


def fetch(mirror, name, path):
    url = f"{mirror}/{REPO}/resolve/main/{name}"
    tmp = path + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {name}  {pct:5.1f}%  "
                          f"{done/2**20:.0f}/{total/2**20:.0f} MB", end="")
    print()
    os.replace(tmp, path)


def main():
    os.makedirs(DST, exist_ok=True)
    for name in FILES:
        path = os.path.join(DST, name)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"已存在, 跳过: {name}")
            continue
        for mirror in MIRRORS:
            try:
                print(f"下载 {name}  ←  {mirror}")
                fetch(mirror, name, path)
                break
            except Exception as e:
                print(f"  失败: {e}")
        else:
            sys.exit(f"无法下载 {name}, 请检查网络")
    print(f"\n完成, 模型位于 {DST}")


if __name__ == "__main__":
    main()
