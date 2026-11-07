# Snapshot: one_layer_Wo1_S2 (decimal-to-binary-8bit, acc 1.0, exhaustively 256/256; 111k params). Condensed:
# only the parts that differ from the stock file.
# Architecture: SimpleTransformer(vocab, max_seq_len=task.seq_len, n_layers=1, d_ff=[1024], d_model=48, n_heads=3)
# SimpleTransformer builds blocks from the d_ff list; Block(use_attn=(i == 0)).

def write_weights(model, task):
    """Residual layout (d_model=48):
      0-10 token one-hot | 11-22 position one-hot (amp 3)
      23-25 hundreds digit (0-2), 26-35 tens, 36-45 units (gathered by attention)
      46 logit '1' | 47 constant 1 (query source and default '0' logit)
    Attention: head h always attends to prompt position h and copies that digit's one-hot.
    MLP: one unit per (N, j) with bit j of N (MSB first) == 1: 4-way AND of the three digits and
         output position 3+j; writes +4 into the '1' logit. No unit fires -> constant makes '0' win.
    """
    for p in model.parameters():
        nn.init.zeros_(p)
    with torch.no_grad():
        for i in range(11):
            model.token_emb.weight[i, i] = 1.0
        for t in range(model.max_seq_len):
            if t < 12:
                model.pos_emb.weight[t, 11 + t] = 3.0
            model.pos_emb.weight[t, 47] = 1.0
        for b in model.blocks:
            b.ln2.weight.fill_(1.0)
            if b.use_attn:
                b.ln1.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        b0 = model.blocks[0]
        S = 2.0
        dh = model.d_model // model.n_heads
        DIG, NV, LOG1, CONST = [23, 26, 36], [3, 10, 10], 46, 47
        for h in range(3):
            b0.attn.W_q.weight[dh * h + dh - 1, CONST] = S
            b0.attn.W_k.weight[dh * h + dh - 1, 11 + h] = 1.0
            for d in range(NV[h]):
                b0.attn.W_v.weight[dh * h + d, d] = 1.0
                b0.attn.W_o.weight[DIG[h] + d, dh * h + d] = 1.0
        u = 0
        for N in range(256):
            a, bb, c = N // 100, (N // 10) % 10, N % 10
            for j in range(8):
                if (N >> (7 - j)) & 1:
                    for dim in (DIG[0] + a, DIG[1] + bb, DIG[2] + c):
                        b0.mlp.fc1.weight[u, dim] = 1.0
                    b0.mlp.fc1.weight[u, 14 + j] = 2.0
                    b0.mlp.fc1.bias[u] = -13.3
                    b0.mlp.fc2.weight[LOG1, u] = 4.0
                    u += 1
        model.head.weight[0, CONST] = 6.0
        model.head.weight[1, LOG1] = 1.0
