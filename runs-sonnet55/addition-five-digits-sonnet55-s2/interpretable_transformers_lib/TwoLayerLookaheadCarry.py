"""Snapshot: TwoLayerLookaheadCarry (100% on addition-five-digits, d_model=128, 2 layers, 4 heads, d_ff=128).

Hand-written 2-layer adder (answer is written most-significant digit first).
Sequence: a0..a4 '+' b0..b4 '=' r0..r5 (r0 = carry-out digit, r1..r5 = digits).
Residual stream subspaces (balanced one-hot codes so LayerNorm only rescales):
  TOK token one-hot, POS position one-hot, AIN a-digit fetched to b_j,
  S one-hot of s_j=a_j+b_j (0..18), G flag s_j>=10, NINE flag s_j==9,
  SSEL one-hot of s_i for the output digit, CAR carry flag into that digit, OUT output digit.
Layer 1: attention at b_j fetches a_j (key = p-6); MLP = 10x10 lookup of a_j+b_j.
Layer 2: head A fetches own s_i (key p-6; '+' gives s=0 for the carry digit);
         head B fetches carry = G flag of nearest non-9 lower position (sink '=' -> 0);
         MLP = lookup (s, carry) -> (s+carry) mod 10.
(The architecture classes and evaluation harness are identical to interpretable_transformer.py;
 only write_weights and the build_model hyperparameters matter.)
"""
import math
import torch


def build_hparams():
    return dict(d_model=128, n_heads=4, n_layers=2, d_ff=128)


def write_weights(model, task):
    D = model.d_model
    dh = D // model.n_heads
    NP = model.max_seq_len
    TOK, POS = 0, 12
    AIN, S, G, NINE = 30, 40, 59, 60
    SSEL, CAR, OUT = 62, 81, 82
    PLUS = task.stoi["+"]
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for t in range(12):
            model.token_emb.weight[t, TOK + t] = 1.0
            model.token_emb.weight[t, TOK:TOK + 12] -= 1 / 12
        for p in range(NP):
            model.pos_emb.weight[p, POS + p] = 1.0
            model.pos_emb.weight[p, POS:POS + NP] -= 1 / NP
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        def nrm2(*blocks):
            return sum((1 - 1 / n) ** 2 + (n - 1) / n ** 2 for n in blocks)
        s0 = math.sqrt(D / nrm2(12, NP))
        s1 = math.sqrt(D / nrm2(12, NP, 10))
        s2 = math.sqrt(D / (nrm2(12, NP, 10, 19) + 1))
        s3 = math.sqrt(D / (nrm2(12, NP, 10, 19, 19) + 2))

        def set_pos_query(attn, head, E):
            R = torch.zeros(NP, NP)
            for (p, q), v in E.items():
                R[p, q] = v
            R[0] -= R.sum(0)
            for q in range(NP):
                attn.W_q.weight[head * dh + q, POS:POS + NP] = R[:, q]
                attn.W_k.weight[head * dh + q, POS + q] = 1.0

        blk = model.blocks[0]
        set_pos_query(blk.attn, 0, {(p, p - 6): 4.0 for p in range(6, 11)})
        for d in range(10):
            blk.attn.W_v.weight[d, TOK + d] = 1 / s0
            blk.attn.W_o.weight[AIN + d, d] = 1.0
        u = 0
        for a in range(10):
            for b in range(10):
                blk.mlp.fc1.weight[u, AIN + a] = 1.0
                blk.mlp.fc1.weight[u, TOK + b] = 1.0
                blk.mlp.fc1.bias[u] = -1.3 * s1
                g = 1 / (0.5 * s1)
                v = a + b
                blk.mlp.fc2.weight[S:S + 19, u] = -g / 19
                blk.mlp.fc2.weight[S + v, u] += g
                if v >= 10:
                    blk.mlp.fc2.weight[G, u] = g
                if v == 9:
                    blk.mlp.fc2.weight[NINE, u] = g
                u += 1
        blk.mlp.fc1.weight[u, TOK + PLUS] = 1.0
        blk.mlp.fc1.bias[u] = -0.5 * s0
        g = 1 / (0.4 * s0)
        blk.mlp.fc2.weight[S:S + 19, u] = -g / 19
        blk.mlp.fc2.weight[S, u] += g

        blk = model.blocks[1]
        set_pos_query(blk.attn, 0, {(p, p - 6): 4.0 for p in range(11, 17)})
        for d in range(19):
            blk.attn.W_v.weight[d, S + d] = 1 / s2
            blk.attn.W_o.weight[SSEL + d, d] = 1.0
        E = {}
        for p in range(11, 17):
            k = p - 11
            for q in range(NP):
                E[(p, q)] = -12.0
            E[(p, 11)] = 2.0
            for j in range(k, 5):
                E[(p, 6 + j)] = 20.0 - 4.0 * (j - k)
        set_pos_query(blk.attn, 1, E)
        c = 1 * dh + 18
        for p in range(11, 17):
            blk.attn.W_q.weight[c, POS + p] = -40.0
        blk.attn.W_k.weight[c, NINE] = 1.0
        blk.attn.W_v.weight[dh, G] = 1 / s2
        blk.attn.W_o.weight[CAR, dh] = 1.0
        u = 0
        for sv in range(19):
            for cy in range(2):
                blk.mlp.fc1.weight[u, SSEL + sv] = 1.0
                blk.mlp.fc1.weight[u, CAR] = 1.0 if cy else -1.0
                blk.mlp.fc1.bias[u] = -(1.5 if cy else 0.45) * s3
                g = 0.5
                blk.mlp.fc2.weight[OUT:OUT + 10, u] = -g / 10
                blk.mlp.fc2.weight[OUT + (sv + cy) % 10, u] += g
                u += 1
        for d in range(10):
            model.head.weight[d, OUT + d] = 1.0
