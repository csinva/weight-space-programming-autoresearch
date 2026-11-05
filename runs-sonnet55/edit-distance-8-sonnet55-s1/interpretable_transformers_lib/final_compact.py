"""Snapshot final_compact (acc 1.0, ~1.14e3 params). Evaluation harness omitted.
17 blocks with tied weights (last block attention only), d_model=11, 2 heads (d_qk=2, d_v=2), d_ff=16, no LayerNorm.
Only per-layer parameters: the 2-number query bias of the cost head (the anti-diagonal offset t+7)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_qk=2, d_v=2):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, d_qk, bias=True)       # head 0 query (shared): points at position p-1
        self.W_qc = nn.Linear(d_model, d_qk, bias=True)      # head 1 query: weight tied, bias per layer
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        q = torch.cat([self.W_q(x), self.W_qc(x)], dim=-1).view(B, T, H, self.d_qk).transpose(1, 2)
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
    def __init__(self, d_model, n_heads, d_ff, use_mlp=True):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff) if use_mlp else None

    def forward(self, x):
        x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=11, n_heads=2, n_layers=17, d_ff=16):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, use_mlp=(i < n_layers - 1)) for i in range(n_layers)])
        b0 = self.blocks[0]
        for blk in self.blocks[1:]:
            blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o = b0.attn.W_q, b0.attn.W_k, b0.attn.W_v, b0.attn.W_o
            blk.attn.W_qc.weight = b0.attn.W_qc.weight
            if blk.mlp is not None:
                blk.mlp = b0.mlp
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def write_weights(model, task):
    """Block t computes DP anti-diagonal t (position 8+j = column j, comma = column 0).
    Registers: R1=S_{t-1}, R2=S_{t-2}. A=own R1; B,C=R1,R2 of position p-1 (head 0).
    S_t = C + (1-M)*(1 - relu(C-A) - relu(C-B) + relu(2C-A-B-1)) (valid since neighbouring DP cells differ by <=1);
    M = [s1 letter == s2 letter] for the next step, fetched by head 1 pointing at position (t+7)-p.
    Pointers use quadratic scores -beta*(k-target)^2 from features (p, p^2). Virtual cells D[i][j]=j-i (i<0)
    are the initial register values (R1=2j, R2=2j+1); comma holds D[t][0]=t. Readout: logit_d = K*(2dv-d^2)."""
    G2, G, K = 20.0, 16.0 * math.sqrt(2), 1.0
    HB, HV = 2, 2
    L0, PS, PS2, R1, R2, NB1, NB2, TOK, ACT, M, CM = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    stoi = task.stoi
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()
        te, pe = model.token_emb.weight, model.pos_emb.weight
        for ch, idx in stoi.items():
            if ch in "abcd":
                te[idx, L0] = float("abcd".index(ch))
        for p in range(pe.shape[0]):
            pe[p, PS] = float(p)
            pe[p, PS2] = float(p * p)
        for p in range(9, 17):
            j = p - 8
            pe[p, ACT] = 1.0
            pe[p, R1] = 2 * j
            pe[p, R2] = 2 * j + 1
        pe[8, R1] = 0.0
        pe[8, R2] = 1.0
        pe[8, CM] = 1.0

        def attn_nbr(blk):
            a = blk.attn
            a.W_q.weight[0, PS] = 2 * G; a.W_q.bias[0] = -2 * G
            a.W_q.bias[1] = -G
            a.W_k.weight[0, PS] = 1.0; a.W_k.weight[1, PS2] = 1.0
            a.W_v.weight[0, R1] = 1.0
            a.W_v.weight[1, R2] = 1.0
            a.W_o.weight[NB1, 0] = 1.0
            a.W_o.weight[NB2, 1] = 1.0

        def attn_cost(blk, t):
            a = blk.attn
            a.W_qc.weight[0, PS] = -2 * G; a.W_qc.bias[0] = 2 * G * (t + 7)
            a.W_qc.bias[1] = -G
            a.W_k.weight[HB, PS] = 1.0; a.W_k.weight[HB + 1, PS2] = 1.0
            a.W_v.weight[HV, L0] = 1.0
            a.W_o.weight[TOK, HV] = 1.0

        for t in range(1, 17):
            blk = model.blocks[t - 1]
            attn_nbr(blk); attn_cost(blk, t + 1)
            w1, b1, w2, b2 = blk.mlp.fc1.weight, blk.mlp.fc1.bias, blk.mlp.fc2.weight, blk.mlp.fc2.bias
            u = 0
            terms = [({}, 1.0, +1), ({NB2: 1, R1: -1}, 0.0, -1), ({NB2: 1, NB1: -1}, 0.0, -1),
                     ({NB2: 2, R1: -1, NB1: -1}, -1.0, +1)]
            for wd, c0, sgn in terms:
                for d_, v in wd.items():
                    w1[u, d_] = v
                w1[u, ACT] = G2; w1[u, M] = -G2
                b1[u] = c0 - G2
                w2[R1, u] = sgn
                u += 1
            for sgn_d, off, sgn_out in [(1, 0, -1), (1, -1, +1), (-1, 0, -1), (-1, -1, +1)]:
                w1[u, TOK] = sgn_d; w1[u, L0] = -sgn_d; b1[u] = off; w2[M, u] = sgn_out; u += 1
            b2[M] = 1.0
            w1[u, M] = 1; w2[M, u] = -1; u += 1
            w1[u, NB2] = 1; w2[R1, u] = 1; w2[NB2, u] = -1; u += 1
            w1[u, R1] = 1; w1[u, ACT] = G2; b1[u] = -G2; w2[R1, u] = -1; u += 1
            w1[u, CM] = 1; w2[R1, u] = 1; u += 1
            w1[u, R1] = 1; w2[R2, u] = 1; u += 1
            w1[u, R2] = 1; w2[R2, u] = -1; u += 1
            for d_ in [NB1, TOK]:
                w1[u, d_] = 1; w2[d_, u] = -1; u += 1
        attn_nbr(model.blocks[16])
        hd, hb = model.head.weight, model.head.bias
        for ch, idx in stoi.items():
            if ch.isdigit() and int(ch) <= 8:
                d = int(ch)
                hd[idx, NB1] = 2 * d * K
                hb[idx] = -(d ** 2) * K
            else:
                hb[idx] = -100.0


model_shorthand_name = "final_compact"
