"""Snapshot SlimDivisor99 (accuracy 0.995 on 200 samples). Reference design for all DivisorCircuit/SlimDivisor
variants in results/overall_results.csv; variants differ only in the constants N_CAND, USE_NULL_FIX, LN_PIN
(and, for the DivisorCircuit* rows, in the non-slim architecture: d_model=6N, d_ff=18N, attention in all 3 blocks).
Harness (main block / upsert) is identical to interpretable_transformer.py and omitted here.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CAND = 99
USE_NULL_FIX = True
LN_PIN = 2.0e4


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_ff, has_attn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model) if has_attn else None
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if has_attn else None
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=4 * N_CAND + 42, n_heads=6, d_head=N_CAND,
                 d_ffs=(18 * N_CAND, N_CAND, 3 * N_CAND), attn_layers=(True, False, False)):
        super().__init__()
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, d_model
        self.n_heads, self.d_head, self.n_layers = n_heads, d_head, len(d_ffs)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_head, f, a) for f, a in zip(d_ffs, attn_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def write_weights(model, task):
    N = N_CAND
    C = LN_PIN                  # +/-C constant dims pin the LayerNorm scale
    D = model.d_model
    dh = model.d_head
    TOK, POS = 0, 12
    CP, CM, NULL = 28, 29, 30
    SA, SB = 31, 31 + N         # per-d residue sums for a and b
    G = 31 + 2 * N              # [d|a] + [d|b]
    GG = 31 + 3 * N             # [d|a and d|b]
    OUT = 31 + 4 * N            # output digit evidence
    gamma = math.sqrt(2 * C * C / D)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for ln in [model.final_ln] + [m for b in model.blocks for m in (b.ln1, b.ln2) if m is not None]:
            ln.weight.fill_(gamma)
        for t in range(12):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p in range(model.max_seq_len):
            model.pos_emb.weight[p, POS + p] = 1.0
            model.pos_emb.weight[p, CP] = C
            model.pos_emb.weight[p, CM] = -C
        src = [0, 1, 2, 4, 5, 6]
        place = [100, 10, 1, 100, 10, 1]
        att = model.blocks[0].attn
        for i in range(6):
            att.W_q.weight[i * dh, CP] = 1.0 / C
            att.W_k.weight[i * dh, POS + src[i]] = 40.0 * math.sqrt(dh)
            for d in range(1, N + 1):
                for t in range(10):
                    att.W_v.weight[i * dh + d - 1, TOK + t] = (t * place[i]) % d
                att.W_o.weight[(SA if i < 3 else SB) + d - 1, i * dh + d - 1] = 1.0
        mlp = model.blocks[0].mlp
        u = 0
        for d in range(1, N + 1):
            for base in (SA, SB):
                for k in range(0, 3 * (d - 1) + 1, d):
                    for j, coef in ((1, 1.0), (0, -2.0), (-1, 1.0)):
                        mlp.fc1.weight[u, base + d - 1] = 1.0
                        mlp.fc1.bias[u] = j - k
                        mlp.fc2.weight[G + d - 1, u] = coef
                        u += 1
        mlp = model.blocks[1].mlp
        for d in range(1, N + 1):
            mlp.fc1.weight[d - 1, G + d - 1] = 1.0
            mlp.fc1.bias[d - 1] = -1.0
            mlp.fc2.weight[GG + d - 1, d - 1] = 1.0
        mlp = model.blocks[2].mlp
        u = 0
        for d in range(1, N + 1):
            for p, digit in ((7, d // 100), (8, (d // 10) % 10), (9, d % 10)):
                mlp.fc1.weight[u, GG + d - 1] = 1.0
                for d2 in range(d + 1, N + 1):
                    mlp.fc1.weight[u, GG + d2 - 1] = -1.0
                mlp.fc1.weight[u, POS + p] = 1.0
                mlp.fc1.bias[u] = -1.0
                mlp.fc2.weight[OUT + digit, u] = 1.0
                u += 1
        for dg in range(10):
            model.head.weight[dg, OUT + dg] = 1.0
        readers = []
        for b in model.blocks:
            if b.attn is not None:
                readers += [b.attn.W_q.weight, b.attn.W_k.weight, b.attn.W_v.weight]
            readers.append(b.mlp.fc1.weight)
        readers.append(model.head.weight)
        if USE_NULL_FIX:
            for W in readers:
                W[:, NULL] = -W.sum(dim=1)   # LayerNorm subtracts the mean; zero row sums cancel it

model_shorthand_name = "SlimDivisor99"
