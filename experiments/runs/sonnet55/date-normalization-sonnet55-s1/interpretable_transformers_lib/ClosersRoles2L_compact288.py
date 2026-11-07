"""Snapshot ClosersRoles2L_compact288 (200/200, ~0.80M params).
Model definition + write_weights() of interpretable_transformer.py (evaluation harness omitted).
Architecture: 2 blocks, d_model=288, 6 heads (dh=48), d_ff=84, LayerNorms replaced by Identity.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# CausalSelfAttention / MLP / Block / SimpleTransformer are identical to ClosersRoles2L.py except
# SimpleTransformer defaults: d_model=288, n_heads=6, n_layers=2, d_ff=84.


def write_weights(model, task) -> None:
    S = task.stoi
    D = [S[str(d)] for d in range(10)]
    P = 42           # positions
    A = 12.0         # attention sharpness
    DH = model.d_model // model.n_heads
    sq = math.sqrt(DH)

    # ---- residual stream layout -------------------------------------
    names = {}
    cur = [0]
    def alloc(name, n):
        names[name] = cur[0]; cur[0] += n
    alloc("TOK", 43); alloc("POS", P); alloc("ONE", 1)
    alloc("C1", 43); alloc("C2", 43)              # raw token 1 / 2 chars back (letters needed for month names)
    alloc("C3", 11); alloc("C4", 10)              # digit (+ '/' flag for C3) 3 / 4 chars back
    alloc("B", 1)                                 # char at position 4 is '.'
    alloc("RY", 1); alloc("RM", 1); alloc("RD", 1)  # role flags (year/month/day source)
    for k in range(1, 5): alloc(f"YC{k}", 10)     # year closer: digit k chars back
    alloc("YS", 1)                                # year closer has '/' 3 back (2-digit year)
    alloc("MT", 10); alloc("MU", 10); alloc("DT", 10); alloc("DU", 10)  # month/day digits (MT/MU also get name lookup)
    alloc("OUT", 10)                              # digit logits
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
            if k <= 2:      # raw token copy
                for t in range(43):
                    a.W_v.weight[h + t, TOK(t)] = 1
                    a.W_o.weight[n(f"C{k}") + t, h + t] = 1
            else:           # digit one-hot (+ slash flag for C3)
                for d in range(10):
                    a.W_v.weight[h + d, TOK(D[d])] = 1
                    a.W_o.weight[n(f"C{k}") + d, h + d] = 1
                if k == 3:
                    a.W_v.weight[h + 10, TOK(S['/'])] = 1
                    a.W_o.weight[n("C3") + 10, h + 10] = 1
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
            outs = [(RM, 1), (n("MT") + mo // 10, 1), (n("MU") + mo % 10, 1)]
            unit([(n("C2") + S[l1], 1), (n("C1") + S[l2], 1), (TOK(S[l3]), 1)], -2, outs)

        # ===== Layer 2 attention: output positions fetch the flagged sources =====
        a = model.blocks[1].attn
        def role_head(hi, role, vals):
            h = hi * DH
            a.W_q.weight[h, ONE] = A * sq
            a.W_k.weight[h, n(role)] = 1
            for vi, (srcs, dst) in enumerate(vals):
                # srcs: list of (kind, base): 'tok' = raw token slot (digit d at base+D[d]); 'dig' = digit slot (base+d)
                for d in range(10):
                    for kind, base in srcs:
                        a.W_v.weight[h + vi * 10 + d, base + (D[d] if kind == 'tok' else d)] = 1
                    a.W_o.weight[n(dst) + d, h + vi * 10 + d] = 1
        t = lambda k: ('tok', n(f"C{k}"))
        g = lambda k: ('dig', n(f"C{k}"))
        role_head(0, "RY", [([t(1)], "YC1"), ([t(2)], "YC2"), ([g(3)], "YC3"), ([g(4)], "YC4")])
        a.W_v.weight[0 * DH + 40, n("C3") + 10] = 1
        a.W_o.weight[n("YS"), 0 * DH + 40] = 1
        role_head(1, "RM", [([t(2), ('dig', n("MT"))], "MT"), ([t(1), ('dig', n("MU"))], "MU")])
        role_head(2, "RD", [([t(2)], "DT"), ([t(1)], "DU")])

        # ===== Layer 2 MLP: position-gated digit routing =====
        m = model.blocks[1].mlp
        u = [0]
        OUT = n("OUT")
        def route(i, src, extra=()):
            p = 31 + i
            for d in range(10):
                unit([(n(src) + d, 1), (POS(p), 1)] + list(extra), -1, [(OUT + d, 1)])
        YS = n("YS")
        route(0, "YC4", [(YS, -1)])
        unit([(YS, 1), (POS(31), 1)], -1, [(OUT + 2, 1)])
        route(1, "YC3")
        unit([(YS, 1), (POS(32), 1)], -1, [(OUT + 0, 1)])
        route(2, "YC2"); route(3, "YC1")
        route(5, "MT"); route(6, "MU"); route(8, "DT"); route(9, "DU")
        for i, src in ((5, "MT"), (8, "DT")):   # zero-pad a missing tens digit
            unit([(POS(31 + i), 1)] + [(n(src) + d, -1) for d in range(10)], 0, [(OUT + 0, 1)])

        # ===== Output head =====
        for d in range(10):
            model.head.weight[D[d], OUT + d] = 1
        model.head.weight[S['-'], POS(35)] = 10
        model.head.weight[S['-'], POS(38)] = 10


model_shorthand_name = "ClosersRoles2L_compact288"
