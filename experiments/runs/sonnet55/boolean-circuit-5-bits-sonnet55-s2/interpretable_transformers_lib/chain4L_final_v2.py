"""Snapshot: chain4L_final_v2 (100% on boolean-circuit-5-bits, 34.4k params).
Only write_weights is reproduced; build with
    SimpleTransformer(vocab_size=6, max_seq_len=16, d_model=42, n_heads=2, n_layers=4, d_ff=12)
and the rest of interpretable_transformer.py unchanged.

The circuit "x0 o1 x1 o2 x2 o3 x3 o4 x4 =" is evaluated strictly left to right.
Block l=1..3 (at digit position 2l) computes S_l = op_l(S_{l-1}, x_l) with 12 ReLU AND-neurons
(one per (S_prev, op, digit) triple); block 4 at '=' applies the final op directly.
"""
import torch


def write_weights(model, task) -> None:
    A, K, c, a0 = 7.0, 30.0, 40.0, 1.0
    D, OP, POS, CK = 0, 2, 5, 16

    def Sd(l): return 0 if l == 0 else 18 + 2 * (l - 1)
    def OPin(l): return 27 if l == 4 else 24
    def Spv(l): return 30 + 2 * (l - 1)
    DIN, FIN = 38, 40
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
        for i in range(model.max_seq_len):
            if i <= 10:
                model.pos_emb.weight[i, POS + i] = A
            model.pos_emb.weight[i, CK] = K
            model.pos_emb.weight[i, CK + 1] = -K
        for ln in [model.final_ln] + [x for b in model.blocks for x in (b.ln1, b.ln2)]:
            ln.weight.fill_(1.0)

        def head_shift(attn, h, shift, src, dst):
            shifts = [shift] if isinstance(shift, int) else list(shift)
            for j in range(11):
                for sh in shifts:
                    if j + sh <= 10:
                        attn.W_q.weight[h * dh + j, POS + j + sh] = c
                attn.W_k.weight[h * dh + j, POS + j] = 1.0
            for m, (s_, d_) in enumerate(zip(src, dst)):
                attn.W_v.weight[h * dh + m, s_] = len(shifts) * A / a0
                attn.W_o.weight[d_, h * dh + m] = 1.0

        def op_fn(o, s, d):
            return [s & d, s | d, s ^ d][o]

        for l in range(1, 5):
            blk = model.blocks[l - 1]
            last = (l == 4)
            Dsrc, out = (DIN, FIN) if last else (D, Sd(l))
            if last:
                head_shift(blk.attn, 0, (1, 2), [OP, OP + 1, OP + 2, D, D + 1],
                           [OPin(l), OPin(l) + 1, OPin(l) + 2, DIN, DIN + 1])
                head_shift(blk.attn, 1, 3, [Sd(l - 1), Sd(l - 1) + 1], [Spv(l), Spv(l) + 1])
            else:
                if l == 1:
                    head_shift(blk.attn, 0, 1, [OP, OP + 1, OP + 2], [OPin(l), OPin(l) + 1, OPin(l) + 2])
                head_shift(blk.attn, 1, 2, [Sd(l - 1), Sd(l - 1) + 1], [Spv(l), Spv(l) + 1])
            n = 0

            def bit(slot, v):
                return [(slot + v, 0.5), (slot + 1 - v, -0.5)]

            def gate(n, inputs, r, theta, eps):
                for i, w in inputs:
                    blk.mlp.fc1.weight[n, i] = w
                blk.mlp.fc1.bias[n] = -theta
                blk.mlp.fc2.weight[out, n] = (A / eps) * (1 - 2 * r)
                blk.mlp.fc2.weight[out + 1, n] = -(A / eps) * (1 - 2 * r)

            for o in range(3):
                for s_ in range(2):
                    for d in range(2):
                        gate(n, [(OPin(l) + o, 2.0)] + bit(Spv(l), s_) + bit(Dsrc, d), op_fn(o, s_, d), 2.4, 1.4)
                        n += 1
        model.head.weight[st["0"], FIN] = 1.0
        model.head.weight[st["1"], FIN + 1] = 1.0
