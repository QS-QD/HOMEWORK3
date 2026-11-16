# -*- coding: utf-8 -*-
"""
download_data.py — 下载《全唐诗》语料

从 chinese-poetry 仓库拉取 12 个分卷（约 5.6 MB）到 data/raw/
如果 GitHub 直链访问不畅，可设置镜像前缀，例如:
    set HF_ENDPOINT=https://ghproxy.net/https://raw.githubusercontent.com
"""
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

BASE = os.environ.get(
    "CORPUS_BASE",
    "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97")

VOLUMES = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]


def main():
    os.makedirs(RAW, exist_ok=True)
    total = 0
    for v in VOLUMES:
        name = f"poet.tang.{v}.json"
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 10000:
            print(f"  跳过 {name}（已存在）")
            total += 1
            continue
        url = f"{BASE}/{name}"
        try:
            print(f"  下载 {name} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dst)
            size = os.path.getsize(dst) / 1024
            print(f"{size:.0f} KB")
            total += 1
        except Exception as e:
            print(f"失败：{e}")
    print(f"\n共 {total}/{len(VOLUMES)} 个分卷就绪，位于 data/raw/")
    if total < len(VOLUMES):
        print("提示：部分分卷失败时可重新运行本脚本，或手动下载到 data/raw/")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
