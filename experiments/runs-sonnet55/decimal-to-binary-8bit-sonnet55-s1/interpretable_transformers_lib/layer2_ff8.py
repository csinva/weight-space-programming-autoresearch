# Snapshot: layer2_ff8 (decimal-to-binary-8bit, acc 1.0). Condensed: only the parts that differ from the stock file.
# Architecture: 2 blocks, d_model=64, n_heads=4, d_ff=[256, 8]; block 0 = attention + MLP, block 1 = MLP only (no attention params).
# Block(use_attn=False) skips ln1/attn. SimpleTransformer builds blocks from the d_ff list.

def write_weights(model, task):
    """Residual layout (d_model=64):
      0-10 token one-hot | 11-22 position one-hot (amp 3)
      23-32 hundreds digit, 33-42 tens, 43-52 units (gathered) | 53-60 bits of N (MSB first)
      61 unused | 62 logit '1' | 63 constant 1
    """
    for p in model.parameters():
        nn.init.zeros_(p)
    with torch.no_grad():
        for i in range(11):
            model.token_emb.weight[i, i] = 1.0
        for t in range(model.max_seq_len):
            if t < 12:
                model.pos_emb.weight[t, 11 + t] = 3.0
            model.pos_emb.weight[t, 63] = 1.0
        for b in model.blocks:
            b.ln2.weight.fill_(1.0)
            if b.use_attn:
                b.ln1.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)

        b0 = model.blocks[0]
        S = 8.0
        for h in range(3):                       # head h reads prompt position h
            b0.attn.W_q.weight[16 * h + 15, 63] = S
            b0.attn.W_k.weight[16 * h + 15, 11 + h] = 1.0
            for d in range(10):
                b0.attn.W_v.weight[16 * h + d, d] = 1.0
                b0.attn.W_o.weight[23 + 10 * h + d, 16 * h + d] = 2.0
        for N in range(256):                     # AND of three digits -> bits of N
            a, bb, c = N // 100, (N // 10) % 10, N % 10
            for dim in (23 + a, 33 + bb, 43 + c):
                b0.mlp.fc1.weight[N, dim] = 1.0
            b0.mlp.fc1.bias[N] = -11.0
            for k in range(8):
                if (N >> (7 - k)) & 1:
                    b0.mlp.fc2.weight[53 + k, N] = 0.9
        b1 = model.blocks[1]
        for j in range(8):                       # unit j: bit j AND position 3+j
            b1.mlp.fc1.weight[j, 53 + j] = 2.0
            b1.mlp.fc1.weight[j, 14 + j] = 1.0
            b1.mlp.fc1.bias[j] = -3.2
            b1.mlp.fc2.weight[62, j] = 4.0
        model.head.weight[0, 63] = 3.0           # default '0'
        model.head.weight[1, 62] = 1.0
