# -*- coding: utf-8 -*-
"""model.py — 字符级 LSTM 语言模型(单层 LSTM, 约 280 万参数)"""
import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """
    结构: Embedding -> LSTM -> Linear
    输入: (B, T) 的字符编号
    输出: (B, T, V) 每个位置对下一个字符的预测分数
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=1, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # 遗忘门偏置置 1, 缓解梯度消失(标准技巧)
                n = p.size(0)
                p.data[n // 4: n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, hidden=None):
        emb = self.drop(self.embed(x))          # (B,T,E)
        out, hidden = self.lstm(emb, hidden)    # (B,T,H)
        logits = self.fc(self.drop(out))        # (B,T,V)
        return logits, hidden

    def param_breakdown(self):
        """返回各子模块参数量, 供报告画参数构成图"""
        def n(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "embed": n(self.embed),
            "lstm": n(self.lstm),
            "fc": n(self.fc),
            "total": n(self),
        }

    @torch.no_grad()
    def generate(self, idx, n_chars, temperature=1.0, top_k=None, eos_id=None):
        """
        自回归生成。idx: (1,T) 起始上下文
        返回生成的字符编号列表
        """
        self.eval()
        hidden = None
        out = []
        for _ in range(n_chars):
            logits, hidden = self(idx[:, -1:], hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)   # (1,1)
            if eos_id is not None and nxt.item() == eos_id:
                break
            out.append(nxt.item())
            idx = torch.cat([idx[:, -256:], nxt], dim=1)   # 截断上下文, 控制显存
        return out
