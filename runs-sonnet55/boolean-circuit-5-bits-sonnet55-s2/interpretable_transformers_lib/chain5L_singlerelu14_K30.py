"""Snapshot: chain5L_singlerelu14_K30 (100%). Only write_weights shown; architecture is
SimpleTransformer(d_model=50, n_heads=2, n_layers=5, d_ff=14), rest of file unchanged.
Left-to-right chain: block l computes S_l = op(S_{l-1}, digit) at each digit position.
"""

def write_weights(model, task) -> None:
    import torch
    A, K, c, a0 = 7.0, 30.0, 75.0, 1.0
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
                for i, w in inputs:
                    blk.mlp.fc1.weight[n, i] = w
                blk.mlp.fc1.bias[n] = -theta
                blk.mlp.fc2.weight[Sd(l), n] = (A / eps) * (1 - 2 * r)
                blk.mlp.fc2.weight[Sd(l) + 1, n] = -(A / eps) * (1 - 2 * r)

            for o in range(3):
                for s_ in range(2):
                    for d in range(2):
                        gate(n, [(OPin(l) + o, 2.0)] + bit(Spv(l), s_) + bit(D, d), op_fn(o, s_, d), 2.4, 1.2)
                        n += 1
            for s_ in range(2):
                gate(n, bit(Spv(l), s_) + [(POS + 0, 1.5)], s_, 1.4, 1.0)
                n += 1
        head_shift(model.blocks[4].attn, 0, 1, [Sd(4), Sd(4) + 1], [FIN, FIN + 1])
        model.head.weight[st["0"], FIN] = 1.0
        model.head.weight[st["1"], FIN + 1] = 1.0
