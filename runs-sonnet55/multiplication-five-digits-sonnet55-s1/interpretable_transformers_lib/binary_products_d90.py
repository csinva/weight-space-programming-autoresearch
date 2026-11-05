"""Snapshot binary_products_d90 (100% acc). Full copy of the model-defining part of interpretable_transformer.py
(harness below the model section is unchanged from the original file and omitted).
Architecture classes (CausalSelfAttention/MLP/Block/SimpleTransformer) are exactly as in the run folder: no LayerNorm,
attention only in the blocks listed in attn_layers, per-block d_ff, plain residual adds.
"""

D, NH = 90, 10
DH = D // NH
NMAX = [8, 17, 26, 35, 44, 36, 27, 18, 9]  # max carry out of each column stage
TOK, POS, CONST = 0, 12, 34
A0, B0, C0, CY0, IND0, SEL = 35, 55, 60, 69, 79, 89
S = 200.0
G = 5  # columns of carry resolved per MLP layer


def build_model(task):
    L = max(task.seq_len, 16)
    ffs = [100] + [2 * sum(NMAX[k0:k0 + G]) for k0 in range(0, 9, G)] + [10, 30]
    model = SimpleTransformer(task.vocab_size, L, D, NH, ffs, attn_layers={0})
    write_weights(model, task)
    model.eval()
    return model


def write_weights(model, task):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        L = model.max_seq_len
        for t in range(task.vocab_size):
            model.token_emb.weight[t, TOK + t] = 1
        for p_ in range(L):
            model.pos_emb.weight[p_, POS + p_] = 1
        model.pos_emb.weight[:, CONST] = 1
        # layer 0 attention: head h looks at a fixed position (a_i at i, b_j at 6+j)
        at = model.blocks[0].attn
        for h in range(10):
            target = h if h < 5 else 6 + (h - 5)
            r = DH * h
            at.W_q.weight[r, CONST] = S
            at.W_k.weight[r, POS + target] = 1
            if h < 5:
                for t in range(4):  # copy the 4 binary digits of a_i
                    for v in range(10):
                        at.W_v.weight[r + t, TOK + v] = (v >> t) & 1
                    at.W_o.weight[A0 + 4 * h + t, r + t] = 1
            else:
                for v in range(10):
                    at.W_v.weight[r, TOK + v] = v
                at.W_o.weight[B0 + (h - 5), r] = 1
        # layer 0 MLP: unit (i,j,t) = 2^t*b_j if bit_t(a_i) else 0
        m = model.blocks[0].mlp
        u = 0
        for i in range(5):
            for j in range(5):
                for t in range(4):
                    x = 2 ** t
                    m.fc1.weight[u, B0 + j] = x
                    m.fc1.weight[u, A0 + 4 * i + t] = 9 * x
                    m.fc1.bias[u] = -9 * x
                    m.fc2.weight[C0 + 8 - i - j, u] = 1  # operands are written MSB first
                    u += 1
        # carry layers: carry into k0+j = floor((cy_k0 + sum_{i<j} c_{k0+i} 10^i)/10^j) as a sum of unit steps
        for gi, k0 in enumerate(range(0, 9, G)):
            m = model.blocks[1 + gi].mlp
            u = 0
            for j in range(1, min(G, 9 - k0) + 1):
                for n in range(1, NMAX[k0 + j - 1] + 1):
                    for uu, bias in ((u, 1 - n * 10 ** j), (u + 1, -n * 10 ** j)):
                        m.fc1.weight[uu, CY0 + k0] = 1
                        for i in range(j):
                            m.fc1.weight[uu, C0 + k0 + i] = 10 ** i
                        m.fc1.bias[uu] = bias
                    m.fc2.weight[CY0 + k0 + j, u] = 1
                    m.fc2.weight[CY0 + k0 + j, u + 1] = -1
                    u += 2
        # readout: select digit scalar by position (digit dm emitted at position 20-dm), then one-hot via triangles
        m = model.blocks[-2].mlp
        for dm in range(10):
            if dm < 9:
                m.fc1.weight[dm, C0 + dm] = 1
                m.fc1.weight[dm, CY0 + dm] = 1
                m.fc1.weight[dm, CY0 + dm + 1] = -10
            else:
                m.fc1.weight[dm, CY0 + 9] = 1
            m.fc1.weight[dm, POS + 20 - dm] = 10
            m.fc1.bias[dm] = -10
            m.fc2.weight[SEL, dm] = 1
        m = model.blocks[-1].mlp
        u = 0
        for v in range(10):
            for off, coef in ((1, 1), (0, -2), (-1, 1)):
                m.fc1.weight[u, SEL] = 1
                m.fc1.bias[u] = -v + off
                m.fc2.weight[IND0 + v, u] = coef
                u += 1
        for v in range(10):
            model.head.weight[task.stoi[str(v)], IND0 + v] = 1
