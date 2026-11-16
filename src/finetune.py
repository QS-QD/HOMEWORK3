# -*- coding: utf-8 -*-
"""
finetune.py — 在《全唐诗》上继续微调 uer/gpt2-chinese-poem

这不是从零训练: 权重来自 Hugging Face 上已在大量古诗上预训练过的 GPT-2,
我们只做小学习率的继续预训练(continue pretraining), 让它适应本语料的风格。

用法:
    python src/finetune.py --eval_only              # 只评估原始模型(baseline)
    python src/finetune.py --epochs 5 --tag ft      # 微调
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "gpt2-chinese-poem")
DATA_DIR = os.path.join(ROOT, "data", "gpt2")
OUT_DIR = os.path.join(ROOT, "outputs")

# 生成用的固定开头, 训练过程中定期采样, 观察能力变化
SEEDS = ["春风", "孤舟", "明月", "故人"]


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
class PoemBlocks(torch.utils.data.Dataset):
    """把连续文本切成定长块, 每块作为一条自回归样本"""

    def __init__(self, ids, block_size, stride=None):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size
        stride = stride or block_size
        n = max(1, (len(self.ids) - 1) // stride)
        self.starts = [i * stride for i in range(n)]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        chunk = self.ids[s:s + self.block_size + 1]
        if len(chunk) < self.block_size + 1:  # 末尾补齐到定长
            pad = self.block_size + 1 - len(chunk)
            chunk = torch.cat([chunk, torch.full((pad,), 0, dtype=torch.long)])
        return chunk


def encode_file(tokenizer, path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None):
    model.eval()
    total_loss, total_tok, correct = 0.0, 0, 0
    for x in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(input_ids=x[:, :-1], labels=x[:, :-1])
        # 手算准确率: 直接取 argmax, 不复制 float32 的完整 logits(省显存)
        pred = out.logits.argmax(-1)
        correct += (pred[:, :-1] == x[:, 1:-1]).sum().item()
        total_tok += x[:, :-1].numel() - x.size(0)  # 减去每行最后一个无法预测的位置
        total_loss += out.loss.item() * (x.size(0) * (x.size(1) - 1))
        del out, pred
    n = sum(x.size(0) * (x.size(1) - 1) for x in loader)
    loss = total_loss / n
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20)),
        "acc": correct / total_tok,
    }


# --------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------
@torch.no_grad()
def sample_text(model, tokenizer, seed, max_new=64, temperature=0.9, top_p=0.9, device="cuda"):
    ids = tokenizer.encode(seed, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new,
        # 不给下限时, 模型偶尔写完两个字就吐 [SEP], 样本短得看不出能力
        min_new_tokens=44,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    new_ids = out[0][len(ids):].tolist()
    # BertTokenizer 的 decode 会用空格把中文字连起来, 这里逐字拼
    text = "".join(tokenizer.convert_ids_to_tokens(i) for i in new_ids)
    text = text.replace("[SEP]", "\n").replace("[PAD]", "").replace("[UNK]", "□")
    return text


def take_samples(model, tokenizer, device, temp=0.9):
    """采一组样本

    强制 min_new_tokens 之后, 模型偶尔会用标点接茬("春风，翠幕…"),
    视觉上不好看, 遇到这种就重采, 最多试 4 次。
    """
    out = {}
    for s in SEEDS:
        txt = sample_text(model, tokenizer, s, 64, temp, 0.9, device)
        for _ in range(3):
            if txt and txt[0] not in "，。！？、；：":
                break
            txt = sample_text(model, tokenizer, s, 64, temp, 0.9, device)
        out[s] = txt
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--block", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--tag", type=str, default="ft")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--local_files_only", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None
    print(f"设备: {device}  混合精度: {amp_dtype}", flush=True)

    from transformers import BertTokenizer, GPT2LMHeadModel

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    print(f"分词器: BertTokenizer, 词表 {tokenizer.vocab_size}", flush=True)

    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"模型: GPT-2 12 层 / 768 维, 参数 {n_param:,}", flush=True)

    # ---- 数据 ----
    train_ids = encode_file(tokenizer, os.path.join(DATA_DIR, "train.txt"))
    val_ids = encode_file(tokenizer, os.path.join(DATA_DIR, "val.txt"))
    train_ds = PoemBlocks(train_ids, args.block)
    val_ds = PoemBlocks(val_ids, args.block)
    print(f"token: 训练 {len(train_ids):,} / 验证 {len(val_ids):,}", flush=True)
    print(f"块数: 训练 {len(train_ds)} / 验证 {len(val_ds)}", flush=True)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    # ---- baseline(未微调)评估: 微调前的原始模型在原语料上什么水平 ----
    t0 = time.time()
    base = evaluate(model, val_dl, device, amp_dtype)
    base["params"] = n_param
    print(f"[baseline] 未微调: loss {base['loss']:.3f}  ppl {base['ppl']:.2f}  "
          f"acc {base['acc']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "base_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)
    if args.eval_only:
        print("baseline 样本:")
        for k, v in take_samples(model, tokenizer, device).items():
            print(f"  {k} -> {v.replace(chr(10), '｜')[:60]}")
        return

    # ---- 优化器 ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    steps_per_epoch = len(train_dl) // args.accum
    total_steps = steps_per_epoch * args.epochs
    warm_steps = int(total_steps * args.warmup)

    def lr_lambda(step):
        if step < warm_steps:
            return step / max(1, warm_steps)
        prog = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history, steps_log = [], []
    samples = []
    best = {"val_loss": 1e9, "epoch": 0}
    gstep = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_n, t_ep = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)

        for it, x in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids=x[:, :-1], labels=x[:, :-1])
                loss = out.loss / args.accum
            loss.backward()
            run_loss += out.loss.item() * x.size(0)
            run_n += x.size(0)
            loss_val = out.loss.item()
            del out, loss

            if (it + 1) % args.accum == 0:
                gn = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                steps_log.append({
                    "step": gstep,
                    "loss": loss_val,
                    "lr": sched.get_last_lr()[0],
                    "grad_norm": float(gn),
                })

        tr_loss = run_loss / run_n
        val = evaluate(model, val_dl, device, amp_dtype)
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val["loss"],
            "val_ppl": val["ppl"],
            "val_acc": val["acc"],
            "time": round(time.time() - t_ep, 1),
        })
        flag = ""
        if val["loss"] < best["val_loss"]:
            best = {"val_loss": val["loss"], "epoch": epoch}
            flag = " *best"
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{args.tag}_best.pt"))
        print(f"epoch {epoch:2d}/{args.epochs} | train {tr_loss:.4f} | val {val['loss']:.4f} "
              f"| ppl {val['ppl']:.2f} | acc {val['acc']:.3f} | {time.time()-t_ep:.0f}s{flag}", flush=True)

        samples.append({"epoch": epoch, **take_samples(model, tokenizer, device)})

    total_time = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{args.tag}_last.pt"))

    metrics = {
        "tag": args.tag,
        "params": {"total": n_param},
        "config": {k: v for k, v in vars(args).items()},
        "history": history,
        "steps": steps_log,
        "samples": samples,
        "best": best,
        "total_time": round(total_time, 1),
        "train_blocks": len(train_ds),
        "val_blocks": len(val_ds),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "baseline": base,
    }
    with open(os.path.join(OUT_DIR, f"{args.tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"完成: {total_time/60:.1f} 分钟, 最佳第 {best['epoch']} 轮, val_loss {best['val_loss']:.4f}")


if __name__ == "__main__":
    main()
