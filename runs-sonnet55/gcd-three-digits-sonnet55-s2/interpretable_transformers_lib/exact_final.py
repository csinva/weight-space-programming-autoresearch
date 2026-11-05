"""Snapshot exact_final (200/200, 1.76e7 params). The evaluation harness is identical to interpretable_transformer.py.
Model + weights (copied from the run file):"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

C_MAG = 1000.0
N_ID = 1000
N_CAND = 1000


class Layout:
    def __init__(self, vocab, maxlen):
        n = 0
        for name, k in (("tok", vocab), ("pos", maxlen), ("const", 1), ("slot", 60),
                        ("A", N_ID), ("H", N_CAND), ("g", 30), ("out", 10)):
            setattr(self, name, n)
            n += k
        self.B = self.A   # a and b share one count-vector N
        self.n_feat = n
        self.bal = n      # balance dim = minus the sum of all other dims (zero-mean residual)
        self.d_model = n + 1
        self.sigma = C_MAG * math.sqrt(2.0 / self.d_model)


class Attn(nn.Module):
    def __init__(self, d_model, n_heads, dk, dv):
        super().__init__()
        self.n_heads, self.dk, self.dv = n_heads, dk, dv
        self.W_q = nn.Linear(d_model, n_heads * dk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * dk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * dv, bias=False)
        self.W_o = nn.Linear(n_heads * dv, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        q = self.W_q(x).view(B, T, H, self.dk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.dk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.dv).transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        sc = sc.masked_fill(mask, float("-inf"))
        return self.W_o((sc.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, -1))


class Block(nn.Module):
    def __init__(self, d_model, d_ff, attn=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=16):
        super().__init__()
        self.lay = Layout(vocab_size, max_seq_len)
        D = self.lay.d_model
        self.token_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)
        self.blocks = nn.ModuleList([Block(D, 2 * N_ID, Attn(D, 6, 1, 10)), Block(D, N_CAND),
                                     Block(D, N_CAND), Block(D, 30)])
        self.final_ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.final_ln(h))


def write_weights(model, task):
    L = model.lay
    sg = L.sigma
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        dig = [task.stoi[str(d)] for d in range(10)]

        def wr(W, f, col, v=1.0):
            W[f, col] += v
            W[L.bal, col] -= v

        def rd(W, row, f, coef=1.0, mag=1.0):
            W[row, f] += coef * sg / mag

        te, pe = model.token_emb.weight, model.pos_emb.weight
        for t in range(te.shape[0]):
            te[t, L.tok + t] = 1
            te[t, L.const] = C_MAG
            te[t, L.bal] = -(1 + C_MAG)
        for p in range(pe.shape[0]):
            pe[p, L.pos + p] = 1
            pe[p, L.bal] = -1
        # L1 attention: head h reads fixed position src[h], copies digit into slot h
        blk = model.blocks[0]
        at = blk.attn
        for h, ps in enumerate([0, 1, 2, 4, 5, 6]):
            rd(at.W_q.weight, h, L.const, coef=10.0, mag=C_MAG)
            rd(at.W_k.weight, h, L.pos + ps, coef=10.0)
            for d in range(10):
                rd(at.W_v.weight, h * 10 + d, L.tok + dig[d])
                wr(at.W_o.weight, L.slot + h * 10 + d, h * 10 + d)
        # L1 MLP: one-hot of a and b (AND of 3 digit slots) into shared N
        W1, b1, W2 = blk.fc1.weight, blk.fc1.bias, blk.fc2.weight
        for which, off in ((0, 0), (1, 3)):
            for m in range(N_ID):
                u = which * N_ID + m
                for j, dg in enumerate((m // 100, (m // 10) % 10, m % 10)):
                    rd(W1, u, L.slot + (off + j) * 10 + dg)
                b1[u] = -2.0
                wr(W2, L.A + m, u)
        # L2: H[d] = relu(sum_{m: d|m} N[m] - 1)  (d divides both a and b)
        blk = model.blocks[1]
        W1, b1, W2 = blk.fc1.weight, blk.fc1.bias, blk.fc2.weight
        for d in range(1, N_CAND):
            for m in range(0, N_ID, d):
                rd(W1, d, L.A + m)
            b1[d] = -1.0
            wr(W2, L.H + d, d)
        # L3: d is the gcd iff H[d] and no multiple k*d has H; write its three digits
        blk = model.blocks[2]
        W1, b1, W2 = blk.fc1.weight, blk.fc1.bias, blk.fc2.weight
        for d in range(1, N_CAND):
            rd(W1, d, L.H + d)
            for p in range(2, N_ID):
                if p * d < N_CAND:
                    rd(W1, d, L.H + p * d, coef=-1.0)
            for k, dg in enumerate((d // 100, (d // 10) % 10, d % 10)):
                wr(W2, L.g + k * 10 + dg, d)
        # L4: gate digit k by output position 7+k
        blk = model.blocks[3]
        W1, b1, W2 = blk.fc1.weight, blk.fc1.bias, blk.fc2.weight
        for k in range(3):
            for t in range(10):
                u = k * 10 + t
                rd(W1, u, L.g + k * 10 + t)
                rd(W1, u, L.pos + 7 + k)
                b1[u] = -1.5
                wr(W2, L.out + t, u, 2.0)
        for t in range(10):
            rd(model.head.weight, dig[t], L.out + t, coef=10.0)
        for blk in model.blocks:
            blk.ln1.weight.fill_(1)
            blk.ln2.weight.fill_(1)
        model.final_ln.weight.fill_(1)
