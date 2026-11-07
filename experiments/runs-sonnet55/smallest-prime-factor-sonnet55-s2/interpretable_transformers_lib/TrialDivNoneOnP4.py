"""
Snapshot TrialDivNoneOnP4: smallest-prime-factor, 2.33e5 params, 1000/1000 (and 2000/2000 on seed 3).
Complete model + weight code (the unchanged eval harness / argparse main of interpretable_transformer.py is omitted).

Circuit (no training, no LayerNorm):
  L1 attn : heads 0-4 read digit i of N into DIG_i; head 5 at query pos 5+j copies digit j into CV.
  L1 mlp  : for each prime p<=313, s_p = sum_i w_i DIG_i, w_i = u*10^(4-i) mod p (signed, unit u minimising #bumps);
            3 relu neurons per multiple k*p form a bump at s_p = k*p -> D_p = [p | N].
  L2 mlp  : adds F_p = relu(D_p - sum_{q<p} D_q) into channel D_p (D+F == 2 only for the smallest dividing prime);
            NONE (stored in P_4) = relu(1 - sum D); clears DIG_0..4 and VAL.
  L3 mlp  : (p,j) neuron = relu(X_p + P_{5+j} - 2) writes digit j of p into OUT[digit] (OUT channels are the cleared
            DIG/VAL channels and P_0..3); gated bump neurons copy digit CV when NONE=1. Digit 0 is the default logit.
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
    add("VAL", 1); add("P", 10)
    add("DIG", 5); add("CV", 1); add("D", len(PRIMES)); names["F"] = names["D"]
    names["NONE"] = names["P"] + 4  # P_0..4 are zero at output positions (>=5)
    names["OUT"] = [names["DIG"] + i for i in range(5)] + [names["VAL"]] + [names["P"] + i for i in range(4)]
    return names, n


LAY, D_MODEL = _layout()


def _bump_neurons():
    out = []
    for pi, p in enumerate(PRIMES):
        best = None
        for u in range(1, p):
            w = []
            for i in range(5):
                r = (u * pow(10, 4 - i, p)) % p
                if r > p // 2:
                    r -= p
                w.append(r)
            lo = 9 * sum(x for x in w if x < 0)
            hi = 9 * sum(x for x in w if x > 0)
            cost = hi // p - (-(-lo // p))
            if best is None or cost < best[0]:
                best = (cost, w, lo, hi)
        _, w, lo, hi = best
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
            Block(d, 3 * len(BUMPS), n_heads=6, d_head=5),
            Block(d, n_p + 1 + 6),
            Block(d, sum(ch != "0" for p in PRIMES for ch in f"{p:05d}") + 27),
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
            model.token_emb.weight[c, L["VAL"]] = float(c)
        for pos in range(model.max_seq_len):
            if pos < 10:
                model.pos_emb.weight[pos, L["P"] + pos] = 1.0

        at = model.blocks[0].attn
        B = 80.0
        dh = at.d_head
        for h in range(5):
            for i in range(10):
                at.W_q.weight[h * dh, L["P"] + i] = 1.0
            at.W_k.weight[h * dh, L["P"] + h] = B
            at.W_v.weight[h * dh, L["VAL"]] = 1.0
            at.W_o.weight[L["DIG"] + h, h * dh] = 1.0
        for j in range(5):
            at.W_q.weight[5 * dh + j, L["P"] + 5 + j] = 1.0
            at.W_k.weight[5 * dh + j, L["P"] + j] = B
        at.W_v.weight[5 * dh, L["VAL"]] = 1.0
        at.W_o.weight[L["CV"], 5 * dh] = 1.0

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
        for q in range(n_p):
            m.fc1.weight[n_p, L["D"] + q] = -1.0
        m.fc1.bias[n_p] = 1.0
        m.fc2.weight[L["NONE"], n_p] = 1.0
        for i, ch in enumerate(L["OUT"][:6]):
            m.fc1.weight[n_p + 1 + i, ch] = 1.0
            m.fc2.weight[ch, n_p + 1 + i] = -1.0

        m = model.blocks[2].mlp
        r = 0
        for pi, p in enumerate(PRIMES):
            digs = f"{p:05d}"
            for j in range(5):
                if digs[j] == "0":
                    continue
                m.fc1.weight[r, L["F"] + pi] = 1.0
                m.fc1.weight[r, L["P"] + 5 + j] = 1.0
                m.fc1.bias[r] = -2.0
                m.fc2.weight[L["OUT"][int(digs[j])], r] = 1.0
                r += 1
        M = 10.0
        for c in range(1, 10):
            for off, coef in ((1, 1.0), (0, -2.0), (-1, 1.0)):
                m.fc1.weight[r, L["CV"]] = 1.0
                m.fc1.weight[r, L["NONE"]] = M
                m.fc1.bias[r] = float(-c + off - M)
                m.fc2.weight[L["OUT"][c], r] = coef
                r += 1

        for c in range(10):
            model.head.weight[c, L["OUT"][c]] = 1.0
        for i in range(5, 10):
            model.head.weight[0, L["P"] + i] = 0.5
