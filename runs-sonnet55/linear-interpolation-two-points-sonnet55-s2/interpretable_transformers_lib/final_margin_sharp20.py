"""
Snapshot of the final model (final_margin_sharp20): 6.07e3 params, d_model 23, 4 blocks, 2000/2000 on seed 3.
Task: linear-interpolation-two-points. Prompt "x1,yy;x2,yy;q=" -> y1 + s*(q-x1), s=(y2-y1)/(x2-x1) in 1..5.
(Harness/eval code below the model is identical to interpretable_transformer.py and is omitted here.)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=True)   # bias = constant query (target position)
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
    def __init__(self, d_model, n_heads, d_ff, has_attn):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads) if has_attn else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=23, n_heads=7, n_layers=4, d_ff=(10, 10, 39, 20)):
        super().__init__()
        self.vocab_size, self.max_seq_len, self.d_model = vocab_size, max_seq_len, d_model
        self.n_heads, self.n_layers, self.d_ff = n_heads, n_layers, d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff[i], i == 0) for i in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


# Residual layout: TOK 0 digit value | P 1 position p | P2 2 p^2 | dx 3, dy 4, q-x1 5 | A 6 answer scalar
#   SD 7..11 slope mismatch |dy - s*dx| | TENS 12 (ones digit A mod 10) | OUT 13..22 digit logits
TOK, P, P2, SLOT, SD, A, TENS, OUT = 0, 1, 2, 3, 7, 6, 12, 13
SRC_POS = [0, 2, 3, 5, 7, 8, 10]   # x1, y1 tens, y1 ones, x2, y2 tens, y2 ones, q
X1, Y1T, Y1O, X2, Y2T, Y2O, Q = range(7)
DXD, DYD, TQD = SLOT, SLOT + 1, SLOT + 2
CONTRIB = {
    X1: {DXD: -1.0, TQD: -1.0}, Y1T: {DYD: -10.0, A: 10.0}, Y1O: {DYD: -1.0, A: 1.0},
    X2: {DXD: 1.0}, Y2T: {DYD: 10.0}, Y2O: {DYD: 1.0}, Q: {TQD: 1.0},
}


def write_weights(model: SimpleTransformer, task) -> None:
    D = model.d_model
    dh = model.blocks[0].attn.d_head
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        dig = [task.stoi[str(j)] for j in range(10)]
        for p in range(model.max_seq_len):   # position embedding = (p, p^2)
            model.pos_emb.weight[p, P] = float(p)
            model.pos_emb.weight[p, P2] = float(p * p)
        for j in range(10):          # token embedding = the digit's numeric value
            model.token_emb.weight[dig[j], TOK] = float(j)

        # Block 0 attention: head h matches position SRC_POS[h] via score c*(2*sp*p - p^2) = const - c*(p-sp)^2
        at = model.blocks[0].attn
        for h, sp in enumerate(SRC_POS):
            r = h * dh
            c = 20.0 * math.sqrt(dh)
            at.W_q.bias[r] = 2.0 * sp * c
            at.W_q.bias[r + 1] = -c
            at.W_k.weight[r, P] = 1.0
            at.W_k.weight[r + 1, P2] = 1.0
            at.W_v.weight[r, TOK] = 1.0
            for dst, coef in CONTRIB[h].items():
                at.W_o.weight[dst, r] = coef

        def unitvec(i):
            v = torch.zeros(D); v[i] = 1.0
            return v

        DX, DY, TQ = unitvec(DXD), unitvec(DYD), unitvec(TQD)

        def unit(mlp, i, w, b, outs):
            mlp.fc1.weight[i] = w
            mlp.fc1.bias[i] = b
            for d, c in outs:
                mlp.fc2.weight[d, i] = c

        # Block 0 MLP: SD[s-1] = |dy - s*dx|  (zero only for the true slope)
        m = model.blocks[0].mlp
        i = 0
        for s in range(1, 6):
            L = DY - s * DX
            for sgn in (1.0, -1.0):
                unit(m, i, sgn * L, 0.0, [(SD + s - 1, 1.0)]); i += 1

        # Block 1 MLP: A += sum_s [mismatch_s == 0] * s*(q-x1)   (A starts at y1)
        m = model.blocks[1].mlp
        M = 46.0
        i = 0
        for s in range(1, 6):
            for sign in (1.0, -1.0):
                w = sign * s * TQ
                w[SD + s - 1] = -M
                unit(m, i, w, 0.0, [(A, sign)]); i += 1

        # Block 2 MLP: ones digit = A - 10*sum_{k=1..9}[A >= 10k]; plus tens readout from A at position 11
        m = model.blocks[2].mlp
        i = 0
        w = torch.zeros(D); w[A] = 1.0
        unit(m, i, w, 0.0, [(TENS, 1.0)]); i += 1
        for k in range(1, 10):
            unit(m, i, w, float(1 - 10 * k), [(TENS, -10.0)]); i += 1
            unit(m, i, w, float(-10 * k), [(TENS, 10.0)]); i += 1
        G = 100.0
        for k in range(10):
            for sgn, lo in ((1.0, 10 * k + 9), (-1.0, 10 * k)):
                w = torch.zeros(D); w[A] = sgn; w[P] = -G
                unit(m, i, w, -sgn * lo + 11 * G, [(OUT + k, -1.0)]); i += 1

        # Block 3 MLP: ones readout logit_k = -|ones - k| gated to position 12
        m = model.blocks[3].mlp
        O = torch.zeros(D); O[TENS] = 1.0
        i = 0
        for k in range(10):
            w = O.clone(); w[P] = G
            unit(m, i, w, -k - 12 * G, [(OUT + k, -1.0)]); i += 1
            w = -O; w[P] = G
            unit(m, i, w, k - 12 * G, [(OUT + k, -1.0)]); i += 1

        # +0.5 margin so the right digit beats non-digit tokens (logit 0)
        for k in range(10):
            model.blocks[3].mlp.fc2.bias[OUT + k] = 0.5
            model.head.weight[dig[k], OUT + k] = 1.0


model_shorthand_name = "final_margin_sharp20"
