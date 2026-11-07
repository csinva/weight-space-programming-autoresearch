"""
Snapshot TrialDivB80: smallest-prime-factor, 4.65e5 params, 200/200 and 1000/1000.
prompt "64507=" -> answer "00251" (spf zero padded; a prime maps to itself).

Circuit (no training, no LayerNorm):
  L1 attn : heads 0-4 read digit i of N (value) into DIG_i; head 5 at query pos 5+j
            copies the one-hot of input digit j into C_*.
  L1 mlp  : for each prime p<=313, s_p = sum_i (10^(4-i) mod p, signed) * DIG_i ; 3 relu bump neurons per
            multiple k*p fire at s_p = k*p  ->  D_p = [p divides N].
  L2 mlp  : F_p = relu(D_p - sum_{q<p} D_q) (smallest dividing prime; overwrites D in place),
            NONE = relu(1 - sum D).
  L3 mlp  : at output position 5+j, F_p writes digit j of p to OUT (digit 0 is the CONST default);
            if NONE, copy digit j of N.
(The eval harness at the bottom of interpretable_transformer.py is unchanged and omitted here.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)

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
    def __init__(self, d_model, d_ff, n_heads=0, d_head=10):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head) if n_heads > 0 else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(x)
        return x + self.mlp(x)


def _primes(n):
    return [i for i in range(2, n + 1) if all(i % q for q in range(2, int(i ** 0.5) + 1))]


PRIMES = _primes(313)  # every composite <= 99999 has a prime factor <= 313


def _layout():
    names, n = {}, 0

    def add(key, size):
        nonlocal n
        names[key] = n
        n += size
    add("T", 10); add("VAL", 1); add("CONST", 1); add("P", 10)
    add("DIG", 5); add("C", 10); add("D", len(PRIMES)); names["F"] = names["D"]  # F overwrites D in place
    add("NONE", 1); add("OUT", 10)
    return names, n


LAY, D_MODEL = _layout()


def _bump_neurons():
    out = []
    for pi, p in enumerate(PRIMES):
        w = []
        for i in range(5):
            r = pow(10, 4 - i, p)
            if r > p // 2:
                r -= p
            w.append(r)
        lo = 9 * sum(x for x in w if x < 0)
        hi = 9 * sum(x for x in w if x > 0)
        for k in range(-(-lo // p), hi // p + 1):
            out.append((pi, w, k))
    return out


BUMPS = _bump_neurons()


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=16):
        super().__init__()
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len
        d = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        n_p = len(PRIMES)
        self.blocks = nn.ModuleList([
            Block(d, 3 * len(BUMPS), n_heads=6, d_head=10),
            Block(d, 2 * n_p + 1),
            Block(d, sum(ch != "0" for p in PRIMES for ch in f"{p:05d}") + 10),
        ])
        self.head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def write_weights(model, task):
    L = LAY
    n_p = len(PRIMES)
    with torch.no_grad():
        for prm in model.parameters():
            prm.zero_()
        for c in range(10):
            model.token_emb.weight[c, L["T"] + c] = 1.0
            model.token_emb.weight[c, L["VAL"]] = float(c)
        for pos in range(model.max_seq_len):
            if pos < 10:
                model.pos_emb.weight[pos, L["P"] + pos] = 1.0
            model.pos_emb.weight[pos, L["CONST"]] = 1.0

        at = model.blocks[0].attn
        B = 80.0
        for h in range(5):  # head h reads input digit h
            at.W_q.weight[h * 10 + h, L["CONST"]] = 1.0
            at.W_k.weight[h * 10 + h, L["P"] + h] = B
            at.W_v.weight[h * 10, L["VAL"]] = 1.0
            at.W_o.weight[L["DIG"] + h, h * 10] = 1.0
        for j in range(5):  # head 5: query pos 5+j looks at pos j
            at.W_q.weight[50 + j, L["P"] + 5 + j] = 1.0
            at.W_k.weight[50 + j, L["P"] + j] = B
        for c in range(10):
            at.W_v.weight[50 + c, L["T"] + c] = 1.0
            at.W_o.weight[L["C"] + c, 50 + c] = 1.0

        m = model.blocks[0].mlp
        for n, (pi, w, k) in enumerate(BUMPS):
            p = PRIMES[pi]
            for t, (off, coef) in enumerate(((1, 1.0), (0, -2.0), (-1, 1.0))):
                r = 3 * n + t
                for i in range(5):
                    m.fc1.weight[r, L["DIG"] + i] = float(w[i])
                m.fc1.bias[r] = float(-k * p + off)
                m.fc2.weight[L["D"] + pi, r] = coef

        m = model.blocks[1].mlp
        for pi in range(n_p):
            m.fc1.weight[pi, L["D"] + pi] = 1.0
            for q in range(pi):
                m.fc1.weight[pi, L["D"] + q] = -1.0
            m.fc2.weight[L["F"] + pi, pi] = 1.0
            m.fc1.weight[n_p + 1 + pi, L["D"] + pi] = 1.0
            m.fc2.weight[L["D"] + pi, n_p + 1 + pi] = -1.0
        for q in range(n_p):
            m.fc1.weight[n_p, L["D"] + q] = -1.0
        m.fc1.bias[n_p] = 1.0
        m.fc2.weight[L["NONE"], n_p] = 1.0

        m = model.blocks[2].mlp
        r = 0
        for pi, p in enumerate(PRIMES):
            digs = f"{p:05d}"
            for j in range(5):
                if digs[j] == "0":
                    continue
                m.fc1.weight[r, L["F"] + pi] = 1.0
                m.fc1.weight[r, L["P"] + 5 + j] = 1.0
                m.fc1.bias[r] = -1.0
                m.fc2.weight[L["OUT"] + int(digs[j]), r] = 1.0
                r += 1
        for c in range(10):
            m.fc1.weight[r, L["NONE"]] = 1.0
            m.fc1.weight[r, L["C"] + c] = 1.0
            m.fc1.bias[r] = -1.0
            m.fc2.weight[L["OUT"] + c, r] = 1.0
            r += 1

        for c in range(10):
            model.head.weight[c, L["OUT"] + c] = 1.0
        model.head.weight[0, L["CONST"]] = 0.5
