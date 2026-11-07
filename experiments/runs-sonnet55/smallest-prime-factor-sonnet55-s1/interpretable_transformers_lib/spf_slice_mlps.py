"""
Snapshot of interpretable_transformer.py for model `spf_slice_mlps`
(accuracy 1.0000, 200/200, 1.23e7 params, smallest-prime-factor).

Task: "64507=" -> "00251" (zero-padded smallest prime factor; primes map to themselves).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PRIME_MAX = 313   # largest possible smallest-prime-factor of a composite < 100000 (313^2 = 97969)
PRIMES = [p for p in range(2, PRIME_MAX + 1) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
NP = len(PRIMES)  # 65
N_RES = sum(PRIMES)  # one hidden neuron per (prime, residue)

TOK = 0               # 11  token one-hot
CONST = 11            # 1   constant 1 (query for the digit-gathering heads)
POS = 12              # 11  position one-hot
G = 23                # 50  gathered digit one-hots: slot i = digit at position i
D_ = G + 50           # NP  D_p = 1 iff p divides N
LOG = D_ + NP         # 11  output logits
A_ = LOG + 11         # 100 one-hot of first two digits (A)
B_ = A_ + 100         # 1000 one-hot of last three digits (B)  (N = 1000*A + B)
D_MODEL = B_ + 1000


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        d_attn = n_heads * d_head
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_attn, bias=False)
        self.W_o = nn.Linear(d_attn, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class SliceMLP(nn.Module):
    """relu-MLP that reads residual dims [r0, r1) and adds its output to dims [w0, w1)."""

    def __init__(self, d_ff, r0, r1, w0, w1):
        super().__init__()
        self.r0, self.r1, self.w0, self.w1 = r0, r1, w0, w1
        self.fc1 = nn.Linear(r1 - r0, d_ff)
        self.fc2 = nn.Linear(d_ff, w1 - w0)

    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x[..., self.r0:self.r1])))
        return F.pad(out, (self.w0, x.shape[-1] - self.w1))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, mlp, use_attn):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if use_attn else None
        self.mlp = mlp

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=D_MODEL, n_heads=5, d_head=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        mlps = [
            SliceMLP(1100, G, G + 50, A_, B_ + 1000),          # L1: digits -> one-hot(A), one-hot(B)
            SliceMLP(N_RES, A_, B_ + 1000, D_, D_ + NP),       # L2: (A,B) -> divisibility by each prime
            SliceMLP(50 + 2 + 3 * NP, POS, D_ + NP, LOG, LOG + 11),  # L3: smallest divisor -> logits
        ]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_head, mlp, use_attn=(i == 0)) for i, mlp in enumerate(mlps)]
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


@torch.no_grad()
def write_weights(model, task) -> None:
    """Pipeline, computed identically at every answer position (5..9):
      L1 attn : head i copies digit i (token at position i) into slot G+10i.
      L1 MLP  : AND-neurons -> one-hot of A = first 2 digits (100), B = last 3 digits (1000).
      L2 MLP  : neuron (p,r) fires iff 1000A = r and B = -r (mod p), i.e. p | N -> D_p.
      L3 MLP  : prime N -> emit digit j of N; composite -> '0','0',hundreds,tens,units of the
                smallest p with D_p=1 (neuron value D_p - #smaller divisors).
    """
    for prm in model.parameters():
        prm.zero_()
    for v in range(11):
        model.token_emb.weight[v, TOK + v] = 1
    for i in range(11):
        model.pos_emb.weight[i, POS + i] = 1
    model.pos_emb.weight[:, CONST] = 1

    at = model.blocks[0].attn
    for h in range(5):
        at.W_q.weight[h * 10, CONST] = 12.0
        at.W_k.weight[h * 10, POS + h] = 10.0
        for d in range(10):
            at.W_v.weight[h * 10 + d, TOK + d] = 1.0
            at.W_o.weight[G + 10 * h + d, h * 10 + d] = 1.0

    m = model.blocks[0].mlp
    for a in range(10):
        for b in range(10):
            n = 10 * a + b
            m.fc1.weight[n, a] = 1
            m.fc1.weight[n, 10 + b] = 1
            m.fc1.bias[n] = -1
            m.fc2.weight[n, n] = 1
    for x in range(10):
        for y in range(10):
            for z in range(10):
                n = 100 + 100 * x + 10 * y + z
                m.fc1.weight[n, 20 + x] = 1
                m.fc1.weight[n, 30 + y] = 1
                m.fc1.weight[n, 40 + z] = 1
                m.fc1.bias[n] = -2
                m.fc2.weight[100 + 100 * x + 10 * y + z, n] = 1

    m = model.blocks[1].mlp
    base = 0
    ar, br = torch.arange(100), torch.arange(1000)
    for k, p in enumerate(PRIMES):
        m.fc1.weight[base + (1000 * ar) % p, ar] = 1
        m.fc1.weight[base + (-br) % p, 100 + br] = 1
        m.fc1.bias[base:base + p] = -1
        m.fc2.weight[k, base:base + p] = 1
        base += p

    m = model.blocks[2].mlp
    cG, cD = G - POS, D_ - POS
    n = 0
    for j in range(5):
        for v in range(10):
            m.fc1.weight[n, cD:cD + NP] = -1
            m.fc1.weight[n, 5 + j] = 1
            m.fc1.weight[n, cG + 10 * j + v] = 1
            m.fc1.bias[n] = -1
            m.fc2.weight[v, n] = 1
            n += 1
    for j in range(2):
        m.fc1.weight[n, cD:cD + NP] = 1
        m.fc1.weight[n, 5 + j] = 10
        m.fc1.bias[n] = -10
        m.fc2.weight[0, n] = 1
        n += 1
    for k, p in enumerate(PRIMES):
        for j in range(2, 5):
            m.fc1.weight[n, cD:cD + k] = -1
            m.fc1.weight[n, cD + k] = 1
            m.fc1.weight[n, 5 + j] = 1
            m.fc1.bias[n] = -1
            m.fc2.weight[int(f"{p:03d}"[j - 2]), n] = 1
            n += 1

    for v in range(11):
        model.head.weight[v, LOG + v] = 1


model_shorthand_name = "spf_slice_mlps"
