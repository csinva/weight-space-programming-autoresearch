"""Snapshot WavefrontDP17_slimattn (accuracy 1.0, ~1.24e5 params). Evaluation harness omitted.
Same as the main file at that point: 17 blocks, d_model=50, 2 heads (d_qk=19, d_v=4), d_ff=25, no LayerNorm."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_qk=19, d_v=4):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.d_qk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.d_qk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.d_v).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_qk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        a = s.masked_fill(mask, float("-inf")).softmax(-1)
        return self.W_o((a @ v).transpose(1, 2).contiguous().view(B, T, H * self.d_v))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=50, n_heads=2, n_layers=17, d_ff=25):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def write_weights(model, task):
    """Block t-1 computes DP anti-diagonal t (position 8+j = column j):
    S_t = C + (1-match)*(1 - relu(C-A) - relu(C-B) + relu(2C-A-B-1)), A=own S_{t-1}, B,C = S_{t-1},S_{t-2} of p-1
    (head 0 attends p-1), match from head 1 attending position t+7-p. Unit gating by letter/active column."""
    G2, G, K = 200.0, 40.0 * math.sqrt(19), 5.0
    HB, HV = 19, 4
    L0, CONST, POS, SLOT, NB1, NB2, TOK, JJ, ACT = 0, 4, 5, 24, 42, 43, 44, 48, 49
    SL = lambda t: SLOT + t + 1
    stoi = task.stoi
    T = 19
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()
        te, pe = model.token_emb.weight, model.pos_emb.weight
        for ch, idx in stoi.items():
            te[idx, CONST] = 1.0
            if ch in "abcd":
                te[idx, L0 + "abcd".index(ch)] = 1.0
        for p in range(pe.shape[0]):
            pe[p, POS + p] = 1.0
        for p in range(9, 17):
            j = p - 8
            pe[p, JJ] = j
            pe[p, ACT] = 1.0
            for t in range(-1, 17):
                pe[p, SL(t)] = 2 * j - t
        for t in range(0, 17):
            pe[8, SL(t)] = t
        pe[8, SL(-1)] = 1.0

        def attn_nbr(blk, t):
            a = blk.attn
            for p in range(1, T):
                a.W_q.weight[p - 1, POS + p] = G
            for k in range(T):
                a.W_k.weight[k, POS + k] = 1.0
            a.W_v.weight[0, SL(t - 1)] = 1.0
            a.W_v.weight[1, SL(t - 2)] = 1.0
            a.W_o.weight[NB1, 0] = 1.0
            a.W_o.weight[NB2, 1] = 1.0

        def attn_cost(blk, t):
            a = blk.attn
            for p in range(T):
                tg = t + 7 - p
                if 0 <= tg < T:
                    a.W_q.weight[HB + tg, POS + p] = G
            for k in range(T):
                a.W_k.weight[HB + k, POS + k] = 1.0
            for k in range(4):
                a.W_v.weight[HV + k, L0 + k] = 1.0
                a.W_o.weight[TOK + k, HV + k] = 1.0

        for t in range(1, 17):
            blk = model.blocks[t - 1]
            attn_nbr(blk, t); attn_cost(blk, t)
            w1, b1, w2 = blk.mlp.fc1.weight, blk.mlp.fc1.bias, blk.mlp.fc2.weight
            A = SL(t - 1)
            u = 0
            terms = [({}, 1.0, +1), ({NB2: 1, A: -1}, 0.0, -1), ({NB2: 1, NB1: -1}, 0.0, -1),
                     ({NB2: 2, A: -1, NB1: -1}, -1.0, +1)]
            for k in range(4):
                for wd, c0, sgn in terms:
                    for d_, v in wd.items():
                        w1[u, d_] = v
                    w1[u, ACT] = G2; w1[u, L0 + k] = G2; w1[u, TOK + k] = -G2
                    b1[u] = c0 - 2 * G2
                    w2[SL(t), u] = sgn
                    u += 1
            w1[u, NB2] = 1; w1[u, ACT] = G2; b1[u] = -G2; w2[SL(t), u] = 1; u += 1
            w1[u, JJ] = 1; w1[u, ACT] = G2; b1[u] = -G2; w2[SL(t), u] = -2; u += 1
            w1[u, ACT] = 1; w2[SL(t), u] = float(t); u += 1
            for d_ in [NB1, NB2] + [TOK + k for k in range(4)]:
                w1[u, d_] = 1; w2[d_, u] = -1; u += 1
        attn_nbr(model.blocks[16], 17)
        hd = model.head.weight
        for ch, idx in stoi.items():
            if ch.isdigit() and int(ch) <= 8:
                d = int(ch)
                hd[idx, NB1] = 2 * d * K
                hd[idx, CONST] = -(d ** 2) * K
            else:
                hd[idx, CONST] = -10000.0


model_shorthand_name = "WavefrontDP17_slimattn"
