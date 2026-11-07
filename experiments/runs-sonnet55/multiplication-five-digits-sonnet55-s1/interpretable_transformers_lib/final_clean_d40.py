"""Snapshot final_clean_d40: model-defining section of interpretable_transformer.py (the evaluation harness
below it in the run file is unchanged and omitted here). Needs the SimpleTransformer/Block/MLP/
CausalSelfAttention classes from that file (no LayerNorm, attention only in block 0, per-block d_ff).

Circuit: L0 attention reads the 10 operand digits; L0 MLP forms the 25 digit products with binary-gated units
and sums them into 9 column sums; two MLP layers per column count carries (multiples of 50, then of 10); one
layer gates the right digit by position, one layer turns it into logits via shared step units.
Accuracy 100% (200/200, 1000/1000 seed 7, 2000/2000 seed 1), 3.41e4 parameters.
"""

D, NH = 40, 10
DH = D // NH
NMAX = [8, 17, 26, 35, 44, 36, 27, 18, 9]  # largest possible carry out of each column
TOK, POSR, POSC, A0, B0 = 0, 4, 9, 14, 34
C0, IND0, SEL = 14, 23, 32
CYS = [34, 35, 36, 37, 38, 0, 1, 2, 3]
S = 60.0
SPLIT = 50


def cy(k):
    return CYS[k - 1]


def build_model(task):
    L = max(task.seq_len, 16)
    carry_ffs = [f for k in range(9)
                 for f in (2 * (NMAX[k] // (SPLIT // 10)), 2 * min(SPLIT // 10 - 1, NMAX[k])) if f > 0]
    ffs = [100 + 29] + carry_ffs + [10, 18]
    model = SimpleTransformer(task.vocab_size, L, D, NH, ffs, attn_layers={0})
    write_weights(model, task)
    model.eval()
    return model


def write_weights(model, task):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        L = model.max_seq_len
        for v in range(10):  # digit tokens carry their 4-bit code; '*' and '=' are zero vectors
            for t in range(4):
                model.token_emb.weight[task.stoi[str(v)], TOK + t] = (v >> t) & 1
        for p_ in range(L):
            model.pos_emb.weight[p_, POSR + p_ // 5] = 1
            model.pos_emb.weight[p_, POSC + p_ % 5] = 1
        at = model.blocks[0].attn
        for h in range(10):
            target = h if h < 5 else 6 + (h - 5)
            r = DH * h
            for rr in range(5):
                at.W_q.weight[r, POSR + rr] = S
            at.W_k.weight[r, POSR + target // 5] = 1
            at.W_k.weight[r, POSC + target % 5] = 1
            if h < 5:
                for t in range(4):
                    at.W_v.weight[r + t, TOK + t] = 1
                    at.W_o.weight[A0 + 4 * h + t, r + t] = 1
            else:
                for t in range(4):
                    at.W_v.weight[r, TOK + t] = 2 ** t
                at.W_o.weight[B0 + (h - 5), r] = 1
        m = model.blocks[0].mlp
        u = 0
        for i in range(5):
            for j in range(5):
                for t in range(4):
                    x = 2 ** t
                    m.fc1.weight[u, B0 + j] = x
                    m.fc1.weight[u, A0 + 4 * i + t] = 9 * x
                    m.fc1.bias[u] = -9 * x
                    m.fc2.weight[C0 + 8 - i - j, u] = 1
                    u += 1
        for s in list(range(A0, A0 + 20)) + list(range(B0, B0 + 5)) + list(range(TOK, TOK + 4)):
            m.fc1.weight[u, s] = 1
            m.fc2.weight[s, u] = -1
            u += 1
        bi = 1
        for k in range(9):
            for nn_, scale, kind in ((NMAX[k] // (SPLIT // 10), SPLIT, "A"), (min(SPLIT // 10 - 1, NMAX[k]), 10, "B")):
                if nn_ == 0:
                    continue
                m = model.blocks[bi].mlp
                bi += 1
                for n in range(1, nn_ + 1):
                    for q, (uu, bias) in enumerate(((2 * (n - 1), 1 - n * scale), (2 * (n - 1) + 1, -n * scale))):
                        m.fc1.weight[uu, C0 + k] = 1
                        if k > 0:
                            m.fc1.weight[uu, cy(k)] = 1
                        if kind == "B":
                            m.fc1.weight[uu, cy(k + 1)] = -10
                        m.fc1.bias[uu] = bias
                        m.fc2.weight[cy(k + 1), uu] = (SPLIT // 10 if kind == "A" else 1) * (1 - 2 * q)
        m = model.blocks[-2].mlp
        for dm in range(10):
            if dm < 9:
                m.fc1.weight[dm, C0 + dm] = 1
                if dm > 0:
                    m.fc1.weight[dm, cy(dm)] = 1
                m.fc1.weight[dm, cy(dm + 1)] = -10
            else:
                m.fc1.weight[dm, cy(9)] = 1
            m.fc1.weight[dm, POSR + (20 - dm) // 5] = 10
            m.fc1.weight[dm, POSC + (20 - dm) % 5] = 10
            m.fc1.bias[dm] = -20
            m.fc2.weight[SEL, dm] = 1
        m = model.blocks[-1].mlp
        for v in range(1, 10):
            for k, off in enumerate((1, 0)):
                m.fc1.weight[2 * (v - 1) + k, SEL] = 1
                m.fc1.bias[2 * (v - 1) + k] = -v + off
                m.fc2.weight[IND0 + v - 1, 2 * (v - 1) + k] = 1 - 2 * k
        for v in range(10):
            row = model.head.weight[task.stoi[str(v)]]
            if v >= 1:
                row[IND0 + v - 1] = 1
            else:
                row[POSR:POSR + 5] = 1
            if v <= 8:
                row[IND0 + v] = -1
