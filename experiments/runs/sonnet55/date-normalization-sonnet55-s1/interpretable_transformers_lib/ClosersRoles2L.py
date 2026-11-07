"""Snapshot ClosersRoles2L: 2-layer, no-LayerNorm hand-wired date normalizer (200/200).

The model-definition + write_weights() portion of interpretable_transformer.py at that run.
(The unchanged evaluation harness at the bottom of the main file is omitted here.)
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
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


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
        self.ln1 = nn.Identity()  # no LayerNorm: keeps hand-built features exact
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=432, n_heads=9, n_layers=2, d_ff=96):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def write_weights(model: SimpleTransformer, task) -> None:
    S = task.stoi
    D = [S[str(d)] for d in range(10)]
    P = 42           # positions
    A = 30.0         # attention sharpness
    DH = model.d_model // model.n_heads
    sq = math.sqrt(DH)

    # ---- residual stream layout -------------------------------------
    names = {}
    cur = [0]
    def alloc(name, n):
        names[name] = cur[0]; cur[0] += n
    alloc("TOK", 43); alloc("POS", P); alloc("ONE", 1)
    for k in range(1, 5): alloc(f"C{k}", 43)      # token k chars back
    alloc("B", 1)                                 # char at position 4 is '.'
    alloc("RY", 1); alloc("RM", 1); alloc("RD", 1)  # role flags (year/month/day source)
    alloc("MTc", 10); alloc("MUc", 10)            # month digits from name lookup
    for k in range(1, 5): alloc(f"YC{k}", 10)     # year closer: digit k chars back
    alloc("YS", 1)                                # year closer has '/' 3 back (2-digit year)
    alloc("MT", 10); alloc("MU", 10); alloc("DT", 10); alloc("DU", 10)
    alloc("OUT", 43)
    assert cur[0] <= model.d_model, cur[0]
    n = lambda k: names[k]
    TOK = lambda t: n("TOK") + t
    POS = lambda p: n("POS") + p
    ONE = n("ONE")

    for p_ in model.parameters():
        p_.data.zero_()

    with torch.no_grad():
        for t in range(43):
            model.token_emb.weight[t, TOK(t)] = 1
        for p in range(P):
            model.pos_emb.weight[p, POS(p)] = 1
            model.pos_emb.weight[p, ONE] = 1

        # ===== Layer 1 attention =====
        a = model.blocks[0].attn
        # heads 0..3: "look k tokens back" -> copy raw token into C_k
        for k in range(1, 5):
            h = (k - 1) * DH
            for p in range(P):
                j = p - k if p >= k else p       # fallback: self
                a.W_q.weight[h + j, POS(p)] = A * sq
            for j in range(P):
                a.W_k.weight[h + j, POS(j)] = 1
            for t in range(43):
                a.W_v.weight[h + t, TOK(t)] = 1
                a.W_o.weight[n(f"C{k}") + t, h + t] = 1
        # head 4: read token at absolute position 4 -> flag B if it is '.'
        h = 4 * DH
        a.W_q.weight[h, ONE] = A * sq
        a.W_k.weight[h, POS(4)] = 1
        a.W_v.weight[h, TOK(S['.'])] = 1
        a.W_o.weight[n("B"), h] = 1

        # ===== Layer 1 MLP: closer detection / role flags / month-name lookup =====
        m = model.blocks[0].mlp
        u = [0]
        def unit(terms, bias, outs):
            i = u[0]; u[0] += 1
            for idx, w in terms:
                m.fc1.weight[i, idx] = w
            m.fc1.bias[i] = bias
            for idx, w in outs:
                m.fc2.weight[idx, i] = w
        pd = [(n("C1") + d, 1.0) for d in D]            # previous char is a digit
        early = [(POS(1), 1.0), (POS(2), 1.0)]
        ans = [(POS(p), 1.0) for p in range(32, P)]
        RY, RM, RD = n("RY"), n("RM"), n("RD")
        unit([(TOK(S['_']), 1)] + pd + [(n("B"), -1)], -1, [(RY, 1)])          # end of text, not B -> year
        unit([(TOK(S['_']), 1), (n("B"), 1)] + pd, -2, [(RD, 1)])              # end of text, B -> day
        unit([(TOK(S['/']), 1)] + pd + early, -2, [(RM, 1)])                   # early slash -> month
        unit([(TOK(S['/']), 1)] + pd + [(i, -w) for i, w in early], -1, [(RD, 1)])  # late slash -> day
        unit([(TOK(S['.']), 1), (POS(4), 1)] + pd, -2, [(RY, 1)])              # YYYY. -> year
        unit([(TOK(S['.']), 1), (POS(7), 1)] + pd, -2, [(RM, 1)])              # YYYY.MM. -> month
        other = [(TOK(S[c]), 1) for c in " ,-snrt"]
        unit(other + pd + [(i, -2 * w) for i, w in ans], -1, [(RD, 1)])        # named-format day closer
        months = task.MONTHS
        for mi, name in enumerate(months):
            l1, l2, l3 = name[0], name[1], name[2]
            mo = mi + 1
            outs = [(RM, 1), (n("MTc") + mo // 10, 1), (n("MUc") + mo % 10, 1)]
            unit([(n("C2") + S[l1], 1), (n("C1") + S[l2], 1), (TOK(S[l3]), 1)], -2, outs)

        # ===== Layer 2 attention: output positions fetch the flagged sources =====
        a = model.blocks[1].attn
        def role_head(hi, role, vals):
            h = hi * DH
            a.W_q.weight[h, ONE] = A * sq
            a.W_k.weight[h, n(role)] = 1
            for vi, (src_list, dst) in enumerate(vals):
                for d in range(10):
                    for src in src_list:
                        a.W_v.weight[h + vi * 10 + d, src + (D[d] if src in [n(f"C{k}") for k in range(1, 5)] else d)] = 1
                    a.W_o.weight[n(dst) + d, h + vi * 10 + d] = 1
        C = lambda k: n(f"C{k}")
        role_head(0, "RY", [([C(1)], "YC1"), ([C(2)], "YC2"), ([C(3)], "YC3"), ([C(4)], "YC4")])
        a.W_v.weight[0 * DH + 40, C(3) + S['/']] = 1
        a.W_o.weight[n("YS"), 0 * DH + 40] = 1
        role_head(1, "RM", [([C(2), n("MTc")], "MT"), ([C(1), n("MUc")], "MU")])
        role_head(2, "RD", [([C(2)], "DT"), ([C(1)], "DU")])

        # ===== Layer 2 MLP: position-gated digit routing =====
        m = model.blocks[1].mlp
        u = [0]
        OUT = n("OUT")
        def route(i, src, extra=()):
            p = 31 + i
            for d in range(10):
                unit([(n(src) + d, 1), (POS(p), 1)] + list(extra), -1, [(OUT + D[d], 1)])
        YS = n("YS")
        route(0, "YC4", [(YS, -1)])
        unit([(YS, 1), (POS(31), 1)], -1, [(OUT + S['2'], 1)])
        route(1, "YC3")
        unit([(YS, 1), (POS(32), 1)], -1, [(OUT + S['0'], 1)])
        route(2, "YC2"); route(3, "YC1")
        route(5, "MT"); route(6, "MU"); route(8, "DT"); route(9, "DU")
        for i, src in ((5, "MT"), (8, "DT")):   # zero-pad a missing tens digit
            unit([(POS(31 + i), 1)] + [(n(src) + d, -1) for d in range(10)], 0, [(OUT + S['0'], 1)])

        # ===== Output head =====
        for t in range(43):
            model.head.weight[t, OUT + t] = 1
        model.head.weight[S['-'], POS(35)] = 10
        model.head.weight[S['-'], POS(38)] = 10


model_shorthand_name = "ClosersRoles2L"
