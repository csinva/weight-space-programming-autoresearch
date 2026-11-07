"""
Interpretable transformer: Levenshtein distance via an anti-diagonal DP wavefront.
Snapshot WavefrontDP33 (accuracy 1.0, 33 layers, d_model 64, 2 heads, d_ff 16, no LayerNorm).
Evaluation harness omitted in this snapshot; see interpretable_transformer.py for it.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        return self.W_o((attn @ v).transpose(1, 2).contiguous().view(B, T, D))


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
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=64, n_heads=2, n_layers=33, d_ff=16):
        super().__init__()
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, d_model
        self.n_heads, self.n_layers, self.d_ff = n_heads, n_layers, d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def write_weights(model, task):
    """Step t (2 blocks) computes anti-diagonal t of the DP table; position 8+j is column j.
    S_t(j) = min(S_{t-1}(j)+1, S_{t-1}(j-1)+1, S_{t-2}(j-1)+cost), cost via attention to position t+7-p."""
    BIG, G, K = 30.0, 40.0 * math.sqrt(32), 5.0
    L0, CONST, POS, SLOT, NB1, NB2, TOK, X, Y = 0, 4, 5, 24, 42, 43, 44, 48, 49
    SL = lambda t: SLOT + t + 1
    stoi = task.stoi
    T = 19
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()
        te, pe = model.token_emb.weight, model.pos_emb.weight
        for ch, idx in stoi.items():
            te[idx, CONST] = 1.0
            for t in range(-1, 17):
                te[idx, SL(t)] = BIG
            if ch in "abcd":
                te[idx, L0 + "abcd".index(ch)] = 1.0
            if ch == ",":
                te[idx, SL(0)] = 0.0
        for p in range(pe.shape[0]):
            pe[p, POS + p] = 1.0

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
                    a.W_q.weight[32 + tg, POS + p] = G
            for k in range(T):
                a.W_k.weight[32 + k, POS + k] = 1.0
            for k in range(4):
                a.W_v.weight[32 + k, L0 + k] = 1.0
                a.W_o.weight[TOK + k, 32 + k] = 1.0

        for t in range(1, 17):
            a, b = model.blocks[2 * (t - 1)], model.blocks[2 * (t - 1) + 1]
            attn_nbr(a, t); attn_cost(a, t)
            w1, b1, w2, b2 = a.mlp.fc1.weight, a.mlp.fc1.bias, a.mlp.fc2.weight, a.mlp.fc2.bias
            w1[0, SL(t - 1)] = 1
            w1[1, SL(t - 1)] = 1; w1[1, NB1] = -1
            w1[2, NB2] = 1
            for k in range(4):
                w1[3 + k, TOK + k] = 1; w1[3 + k, L0 + k] = 1; b1[3 + k] = -1
                w1[8 + k, TOK + k] = 1
            w1[7, NB1] = 1
            w2[X, 0] = 1; w2[X, 1] = -1; b2[X] = 1
            w2[Y, 2] = 1; b2[Y] = 1
            for k in range(4):
                w2[Y, 3 + k] = -1
                w2[TOK + k, 8 + k] = -1
            w2[NB1, 7] = -1; w2[NB2, 2] = -1
            w1, b1, w2, b2 = b.mlp.fc1.weight, b.mlp.fc1.bias, b.mlp.fc2.weight, b.mlp.fc2.bias
            w1[0, X] = 1; w1[0, Y] = -1
            w1[1, X] = 1; w1[2, Y] = 1
            w2[SL(t), 1] = 1; w2[SL(t), 0] = -1; b2[SL(t)] = -BIG
            w2[X, 1] = -1; w2[Y, 2] = -1
        attn_nbr(model.blocks[32], 17)
        hd = model.head.weight
        for ch, idx in stoi.items():
            if ch.isdigit() and int(ch) <= 8:
                d = int(ch)
                hd[idx, NB1] = 2 * d * K
                hd[idx, CONST] = -(d ** 2) * K
            else:
                hd[idx, CONST] = -10000.0


model_shorthand_name = "WavefrontDP33"
