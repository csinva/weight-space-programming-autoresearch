"""
Snapshot GroupScan4_Final: smallest-prime-factor, 6.30e4 params, 5000/5000 (seed 6), d_model 21, 35 blocks.
Model + weight code only (the unchanged eval harness / argparse main of interpretable_transformer.py is omitted).

"Scan over primes in groups" circuit (no training, no LayerNorm):
  Block 0 attention: heads 0-4 read input digit i into DIG_i; head 5 (query pos 5+j) copies digit j into CV.
  For each group of GROUP=4 consecutive primes: a bump block and a select block (MLP-only).
    bump  : s_p = sum_i w_i DIG_i (w_i = u*10^(4-i) mod p, signed; unit u minimises the number of bumps);
            3 relu neurons per multiple k*p give [s_p == k*p] -> D_gi = [p | N] (shared channels D_0..3).
    select: relu(D_gi - sum_{q<gi} D_q - G) = "first divisor of the group and none found before";
            adds digit_j(p) into A_j (stored in P_0..4), sets G (stored in VAL), and clears D_gi.
  Decode block: gated bump neurons turn the scalar A_j at output position 5+j into a one-hot logit; if G=0
  (N has no prime factor <= 313, so N is prime) the copied digit CV is written instead. Digit 0 = default logit.
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
    def __init__(self, d_model, d_ff, n_heads=0, d_head=5):
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
GROUP = 4
GROUPS = [PRIMES[i:i + GROUP] for i in range(0, len(PRIMES), GROUP)]


def _layout():
    names, n = {}, 0

    def add(key, size):
        nonlocal n
        names[key] = n
        n += size
    add("VAL", 1); add("P", 10); add("DIG", 5); add("CV", 1); add("D", GROUP)
    names["G"] = names["VAL"]   # VAL is cleared by block 0's MLP after attention has read it
    names["A"] = names["P"]     # P_0..4 are zero at output positions (>=5)
    scratch = [names["D"] + i for i in range(GROUP)] + [names["DIG"] + i for i in range(5)]
    extra = max(0, 9 - len(scratch))
    add("OUTX", extra)
    scratch += [names["OUTX"] + i for i in range(extra)]
    names["OUT"] = {c: scratch[c - 1] for c in range(1, 10)}
    return names, n


LAY, D_MODEL = _layout()


def _bumps(p):
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
        ks = list(range(-(-lo // p), hi // p + 1))
        if best is None or len(ks) < len(best[1]):
            best = (w, ks)
    return best


BUMPS = {p: _bumps(p) for p in PRIMES}


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=16):
        super().__init__()
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len
        d = D_MODEL
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        blocks = []
        for t, grp in enumerate(GROUPS):
            n_bump = sum(len(BUMPS[p][1]) for p in grp)
            blocks.append(Block(d, 3 * n_bump + (1 if t == 0 else 0), n_heads=6 if t == 0 else 0))
            last = t == len(GROUPS) - 1
            blocks.append(Block(d, 2 * len(grp) + (5 if last else 0)))
        blocks.append(Block(d, 5 * 9 * 3 + 9 * 3))
        self.blocks = nn.ModuleList(blocks)
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
    BUMP = ((1, 1.0), (0, -2.0), (-1, 1.0))  # relu(x+1) - 2 relu(x) + relu(x-1) == [x == 0] on integers
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

        for t, grp in enumerate(GROUPS):
            m = model.blocks[2 * t].mlp
            r = 0
            for gi, p in enumerate(grp):
                w, ks = BUMPS[p]
                for k in ks:
                    for off, coef in BUMP:
                        for i in range(5):
                            m.fc1.weight[r, L["DIG"] + i] = float(w[i])
                        m.fc1.bias[r] = float(-k * p + off)
                        m.fc2.weight[L["D"] + gi, r] = coef
                        r += 1
            if t == 0:
                m.fc1.weight[r, L["VAL"]] = 1.0
                m.fc2.weight[L["VAL"], r] = -1.0
            m = model.blocks[2 * t + 1].mlp
            for gi, p in enumerate(grp):
                m.fc1.weight[gi, L["D"] + gi] = 1.0
                for q in range(gi):
                    m.fc1.weight[gi, L["D"] + q] = -1.0
                m.fc1.weight[gi, L["G"]] = -1.0
                digs = f"{p:05d}"
                for j in range(5):
                    m.fc2.weight[L["A"] + j, gi] = float(digs[j])
                m.fc2.weight[L["G"], gi] = 1.0
                m.fc1.weight[len(grp) + gi, L["D"] + gi] = 1.0
                m.fc2.weight[L["D"] + gi, len(grp) + gi] = -1.0
            if t == len(GROUPS) - 1:
                for i in range(5):
                    m.fc1.weight[2 * len(grp) + i, L["DIG"] + i] = 1.0
                    m.fc2.weight[L["DIG"] + i, 2 * len(grp) + i] = -1.0

        m = model.blocks[-1].mlp
        M = 10.0
        r = 0
        for j in range(5):
            for c in range(1, 10):
                for off, coef in BUMP:
                    m.fc1.weight[r, L["A"] + j] = 1.0
                    m.fc1.weight[r, L["P"] + 5 + j] = M
                    m.fc1.bias[r] = float(-c + off - M)
                    m.fc2.weight[L["OUT"][c], r] = coef
                    r += 1
        for c in range(1, 10):
            for off, coef in BUMP:
                m.fc1.weight[r, L["CV"]] = 1.0
                m.fc1.weight[r, L["G"]] = -M
                m.fc1.bias[r] = float(-c + off)
                m.fc2.weight[L["OUT"][c], r] = coef
                r += 1

        for c in range(1, 10):
            model.head.weight[c, L["OUT"][c]] = 1.0
        for i in range(5, 10):
            model.head.weight[0, L["P"] + i] = 0.5
