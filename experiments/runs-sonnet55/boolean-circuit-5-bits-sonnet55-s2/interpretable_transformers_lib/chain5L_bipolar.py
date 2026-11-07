"""Snapshot: chain5L_bipolar (100% on boolean-circuit-5-bits). See write_weights below.
Architecture used: SimpleTransformer(d_model=50, n_heads=2, n_layers=5, d_ff=32).
The rest of the file (architecture, harness) is identical to interpretable_transformer.py.
"""

def write_weights(model, task) -> None:
    """Left-to-right evaluation as a chain of 4 sequential steps (one per layer)."""
    import torch
    A, K, c, a0 = 21.0, 100.0, 75.0, 0.95
    D, OP, EQ, POS, CK = 0, 2, 5, 6, 17

    def Sd(l): return 0 if l == 0 else 19 + 2 * (l - 1)
    def OPin(l): return 27 + 3 * (l - 1)
    def Spv(l): return 39 + 2 * (l - 1)
    FIN = 47
    dh = model.d_model // model.n_heads
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        st = task.stoi
        for ch in "01":
            model.token_emb.weight[st[ch], D + int(ch)] = A
            model.token_emb.weight[st[ch], D + 1 - int(ch)] = -A
        for k, ch in enumerate("&|^"):
            model.token_emb.weight[st[ch], OP + k] = A
        model.token_emb.weight[st["="], EQ] = A
        for i in range(model.max_seq_len):
            if i <= 10:
                model.pos_emb.weight[i, POS + i] = A
            model.pos_emb.weight[i, CK] = K
            model.pos_emb.weight[i, CK + 1] = -K
        for ln in [model.final_ln] + [x for b in model.blocks for x in (b.ln1, b.ln2)]:
            ln.weight.fill_(1.0)

        def head_shift(attn, h, shift, src, dst):
            for j in range(11):
                if j + shift <= 10:
                    attn.W_q.weight[h * dh + j, POS + j + shift] = c
                attn.W_k.weight[h * dh + j, POS + j] = 1.0
            for m, (s_, d_) in enumerate(zip(src, dst)):
                attn.W_v.weight[h * dh + m, s_] = A / a0
                attn.W_o.weight[d_, h * dh + m] = 1.0

        def op_fn(o, s, d):
            return [s & d, s | d, s ^ d][o]

        for l in range(1, 5):
            blk = model.blocks[l - 1]
            head_shift(blk.attn, 0, 1, [OP, OP + 1, OP + 2], [OPin(l), OPin(l) + 1, OPin(l) + 2])
            head_shift(blk.attn, 1, 2, [Sd(l - 1), Sd(l - 1) + 1], [Spv(l), Spv(l) + 1])
            n = 0

            def bit(slot, v):
                return [(slot + v, 0.5), (slot + 1 - v, -0.5)]

            def gate(n, inputs, r, theta, eps):
                for k in (n, n + 1):
                    for i, w in inputs:
                        blk.mlp.fc1.weight[k, i] = w
                blk.mlp.fc1.bias[n] = -theta
                blk.mlp.fc1.bias[n + 1] = -theta - eps
                for k, sgn in ((n, 1), (n + 1, -1)):
                    blk.mlp.fc2.weight[Sd(l), k] = sgn * (A / eps) * (1 - 2 * r)
                    blk.mlp.fc2.weight[Sd(l) + 1, k] = -sgn * (A / eps) * (1 - 2 * r)

            for o in range(3):
                for s_ in range(2):
                    for d in range(2):
                        gate(n, [(OPin(l) + o, 2.0)] + bit(Spv(l), s_) + bit(D, d), op_fn(o, s_, d), 2.4, 0.4)
                        n += 2
            for s_ in range(2):
                gate(n, bit(Spv(l), s_) + [(POS + 0, 1.5)], s_, 1.4, 0.4)
                n += 2
        head_shift(model.blocks[4].attn, 0, 1, [Sd(4), Sd(4) + 1], [FIN, FIN + 1])
        model.head.weight[st["0"], FIN] = 1.0
        model.head.weight[st["1"], FIN + 1] = 1.0
