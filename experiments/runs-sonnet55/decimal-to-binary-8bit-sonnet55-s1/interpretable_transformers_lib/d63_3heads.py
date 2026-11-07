# Snapshot: d63_3heads (decimal-to-binary-8bit, acc 1.0). Condensed: only parts that differ from the stock file.
# Architecture: SimpleTransformer(vocab, max_seq_len=task.seq_len, n_layers=2, d_ff=[256, 8], d_model=63, n_heads=3)
# Block(use_attn=False) for block 1 (MLP only). SimpleTransformer builds blocks from the d_ff list.

def write_weights(model, task):
    """Residual layout (d_model=63):
      0-10 token one-hot | 11-22 position one-hot (amp 3)
      23-32 hundreds digit, 33-42 tens, 43-52 units (gathered) | 53-60 bits of N (MSB first) | 61 logit '1'
    """
    for p in model.parameters():
        nn.init.zeros_(p)
    with torch.no_grad():
        for i in range(11):
            model.token_emb.weight[i, i] = 1.0
        for t in range(model.max_seq_len):
            if t < 12:
                model.pos_emb.weight[t, 11 + t] = 3.0
        for b in model.blocks:
            b.ln2.weight.fill_(1.0)
            if b.use_attn:
                b.ln1.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        b0 = model.blocks[0]
        S = 8.0
        dh = model.d_model // model.n_heads
        for h in range(3):                       # head h reads prompt position h
            b0.attn.W_q.weight[dh * h + dh - 1, 11:23] = S / 3.0   # any position one-hot -> constant query
            b0.attn.W_k.weight[dh * h + dh - 1, 11 + h] = 1.0
            for d in range(10):
                b0.attn.W_v.weight[dh * h + d, d] = 1.0
                b0.attn.W_o.weight[23 + 10 * h + d, dh * h + d] = 2.0
        for N in range(256):                     # AND of three digits -> bits of N
            a, bb, c = N // 100, (N // 10) % 10, N % 10
            for dim in (23 + a, 33 + bb, 43 + c):
                b0.mlp.fc1.weight[N, dim] = 1.0
            b0.mlp.fc1.bias[N] = -11.0
            for k in range(8):
                if (N >> (7 - k)) & 1:
                    b0.mlp.fc2.weight[53 + k, N] = 1.4
        b1 = model.blocks[1]
        for j in range(8):                       # unit j: bit j AND position 3+j
            b1.mlp.fc1.weight[j, 53 + j] = 2.0
            b1.mlp.fc1.weight[j, 14 + j] = 1.0
            b1.mlp.fc1.bias[j] = -3.2
            b1.mlp.fc2.weight[61, j] = 4.0
        model.head.weight[0, 14:22] = 1.0        # default '0' at answer positions
        model.head.weight[1, 61] = 1.0
