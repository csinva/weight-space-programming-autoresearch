# Snapshot: d57_const_restored (decimal-to-binary-8bit, acc 1.0, 45.9k params). Condensed: only parts that differ from the stock file.
# Architecture: SimpleTransformer(vocab, max_seq_len=task.seq_len, n_layers=2, d_ff=[256, 8], d_model=57, n_heads=3)
# Block(use_attn=False) for block 1 (MLP only). SimpleTransformer builds blocks from the d_ff list.

def write_weights(model, task):
    """Residual layout (d_model=57):
      0-10 token one-hot | 11-22 position one-hot (amp 3)
      23-25 hundreds digit (0-2), 26-35 tens, 36-45 units (gathered)
      46-53 bits of N (MSB first) | 54 logit '1' | 55 constant 1
    """
    for p in model.parameters():
        nn.init.zeros_(p)
    with torch.no_grad():
        for i in range(11):
            model.token_emb.weight[i, i] = 1.0
        for t in range(model.max_seq_len):
            if t < 12:
                model.pos_emb.weight[t, 11 + t] = 3.0
            model.pos_emb.weight[t, 55] = 1.0
        for b in model.blocks:
            b.ln2.weight.fill_(1.0)
            if b.use_attn:
                b.ln1.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        b0 = model.blocks[0]
        S = 8.0
        dh = model.d_model // model.n_heads
        DIG, NV, BIT, LOG1, CONST = [23, 26, 36], [3, 10, 10], 46, 54, 55
        for h in range(3):                       # head h reads prompt position h
            b0.attn.W_q.weight[dh * h + dh - 1, CONST] = S
            b0.attn.W_k.weight[dh * h + dh - 1, 11 + h] = 1.0
            for d in range(NV[h]):
                b0.attn.W_v.weight[dh * h + d, d] = 1.0
                b0.attn.W_o.weight[DIG[h] + d, dh * h + d] = 2.0
        for N in range(256):                     # AND of three digits -> bits of N
            a, bb, c = N // 100, (N // 10) % 10, N % 10
            for dim in (DIG[0] + a, DIG[1] + bb, DIG[2] + c):
                b0.mlp.fc1.weight[N, dim] = 1.0
            b0.mlp.fc1.bias[N] = -10.0
            for k in range(8):
                if (N >> (7 - k)) & 1:
                    b0.mlp.fc2.weight[BIT + k, N] = 1.4
        b1 = model.blocks[1]
        for j in range(8):                       # unit j: bit j AND position 3+j
            b1.mlp.fc1.weight[j, BIT + j] = 2.0
            b1.mlp.fc1.weight[j, 14 + j] = 1.0
            b1.mlp.fc1.bias[j] = -3.2
            b1.mlp.fc2.weight[LOG1, j] = 4.0
        model.head.weight[0, CONST] = 3.0        # default '0'
        model.head.weight[1, LOG1] = 1.0
