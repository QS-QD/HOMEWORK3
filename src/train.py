# -*- coding: utf-8 -*-
"""
train.py — 训练字符级 LSTM 语言模型

用法:
    python src/train.py                        # 默认配置
    python src/train.py --epochs 20 --tag exp1 # 对比实验
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import CharLSTM  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")

SEEDS = ["春風", "山中", "明月", "行路難"]


def set_seed(s=42):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_data(seq_len, val_ratio=0.03):
    with open(os.path.join(DATA, "poetry.txt"), encoding="utf-8") as f:
        text = f.read()
    with open(os.path.join(DATA, "vocab.json"), encoding="utf-8") as f:
        v = json.load(f)
    vocab = v["vocab"]
    itos = {int(k): s for k, s in v["itos"].items()}

    ids = np.array([vocab[c] for c in text], dtype=np.int32)
    L = seq_len + 1
    n = len(ids) // L
    arr = ids[: n * L].reshape(n, L)

    perm = np.random.RandomState(0).permutation(n)
    n_val = max(8, int(n * val_ratio))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return arr[train_idx], arr[val_idx], vocab, itos


@torch.no_grad()
def evaluate(model, arr, batch_size, crit, device):
    model.eval()
    total, count, correct = 0.0, 0, 0
    for i in range(0, len(arr), batch_size):
        b = torch.from_numpy(arr[i:i + batch_size].astype(np.int64)).to(device)
        x, y = b[:, :-1], b[:, 1:]
        logits, _ = model(x)
        loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item() * y.numel()
        count += y.numel()
        correct += (logits.argmax(-1) == y).sum().item()
    return total / count, correct / count


@torch.no_grad()
def sample(model, itos, vocab, seed_text, n_chars, device, temperature=0.8):
    model.eval()
    idx = torch.tensor([[vocab[c] for c in seed_text]], dtype=torch.long, device=device)
    ids = model.generate(idx, n_chars, temperature=temperature)
    return seed_text + "".join(itos.get(i, "") for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--tag", type=str, default="lstm")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 配置: {vars(args)}")

    train_arr, val_arr, vocab, itos = load_data(args.seq_len)
    V = len(vocab)
    print(f"训练块 {len(train_arr)} / 验证块 {len(val_arr)} / 词表 {V}")

    model = CharLSTM(V, args.embed_dim, args.hidden_dim,
                     args.num_layers, args.dropout).to(device)
    pb = model.param_breakdown()
    print(f"参数量: {pb['total']:,} (embed {pb['embed']:,} / "
          f"lstm {pb['lstm']:,} / fc {pb['fc']:,})")

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_arr) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.15, div_factor=10, final_div_factor=20)

    history, step_log, samples = [], [], []
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_arr))
        ep_loss, nb = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            batch = torch.from_numpy(
                train_arr[order[i:i + args.batch_size]].astype(np.int64)).to(device)
            x, y = batch[:, :-1], batch[:, 1:]

            logits, _ = model(x)
            loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            lr_now = sched.get_last_lr()[0]
            sched.step()

            ep_loss += loss.item()
            nb += 1
            step_log.append({"step": len(step_log) + 1,
                             "loss": round(loss.item(), 4),
                             "grad_norm": round(float(gnorm), 4),
                             "lr": round(lr_now, 6)})

        train_loss = ep_loss / nb
        val_loss, val_acc = evaluate(model, val_arr, args.batch_size, crit, device)
        val_ppl = math.exp(min(val_loss, 20))

        # 每轮采一段, 固定同一个开头, 记录生成能力随训练的变化
        gen = sample(model, itos, vocab, "春風", 52, device, 0.8)
        samples.append({"epoch": epoch, "text": gen.replace("<e>", "｜")})

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "val_ppl": round(val_ppl, 2),
                        "val_acc": round(val_acc, 4),
                        "lr": round(sched.get_last_lr()[0], 6),
                        "elapsed": round(time.time() - t0, 1)})
        print(f"epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | ppl {val_ppl:7.2f} | acc {val_acc:.3f} | "
              f"{time.time() - t0:5.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(OUT, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "vocab_size": V,
                        "config": {"embed_dim": args.embed_dim,
                                   "hidden_dim": args.hidden_dim,
                                   "num_layers": args.num_layers,
                                   "dropout": args.dropout},
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "vocab": vocab}, os.path.join(OUT, f"{args.tag}_best.pt"))

    metrics = {
        "config": vars(args),
        "params": pb,
        "vocab_size": V,
        "train_chunks": int(len(train_arr)),
        "val_chunks": int(len(val_arr)),
        "device": str(device),
        "total_time": round(time.time() - t0, 1),
        "best_val_loss": round(best_val, 4),
        "history": history,
        "steps": step_log,
        "samples": samples,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{args.tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)

    print(f"\n完成, 用时 {metrics['total_time']}s, 最佳验证损失 {best_val:.4f}")
    print("\n最终生成样例:")
    for s in samples[-2:]:
        print(f"[epoch {s['epoch']}] {s['text']}")


if __name__ == "__main__":
    main()
