"""Snapshot: final_d37_ff18_B8 (1 layer, 1 head, d_model=37, d_ff=18, 8.49k params). 200/200 (and 1000/1000 on seed 2).
SimpleTransformer defaults for this snapshot: d_model=37, n_heads=1, n_layers=1, d_ff=18.
Only write_weights is shown; the rest of the file matches interpretable_transformer.py.
"""
import math
import torch


def write_weights(model, task) -> None:
    # Residual layout (d_model=37):
    #   0..9 digit one-hot | 10..15 position one-hot (positions 5..10 only) | 16,17 constants +B,-B-(onehots)
    #   18..27 digit histogram (mean) | 28..36 output dims for digits 0..8 (digit 9 = 1 - sum of them)
    # Constants keep the residual zero-mean, so LayerNorm is ~ a constant rescale undone by its gain.
    # Attention: all positions attend uniformly to the 5 prompt digits -> digit histogram (mean = count/5).
    # MLP: at output step k (from position), I_d = clamp(C_d - k, 0, 1) = [cumcount(<=d) > k];
    #      answer digit = smallest d with I_d = 1, so logit_d = I_d - I_{d-1}.
    D = model.d_model
    B = 8.0
    TOK, POS, C1, C2, HIST, LOG = 0, 10, 16, 17, 18, 28
    for p in model.parameters():
        p.data.zero_()
    dig = [task.stoi[str(d)] for d in range(10)]
    with torch.no_grad():
        for d in range(10):
            model.token_emb.weight[dig[d], TOK + d] = 1.0
        model.token_emb.weight[:, C1] = B
        model.token_emb.weight[:, C2] = -B
        model.token_emb.weight[dig, C2] = -(B + 1)
        for p in range(5, task.seq_len):
            model.pos_emb.weight[p, POS + p - 5] = 1.0
            model.pos_emb.weight[p, C2] = -1.0
        g = math.sqrt((2 + B * B + (B + 1.5) ** 2) / D)
        blk = model.blocks[0]
        for ln in (blk.ln1, blk.ln2, model.final_ln):
            ln.weight.fill_(g)
        blk.attn.W_q.weight[0, C1] = 8.0
        blk.attn.W_k.weight[0, C1] = 1.0 / B
        for p in range(5, 11):
            blk.attn.W_k.weight[0, POS + p - 5] = -1.0
        for d in range(10):
            blk.attn.W_v.weight[HIST + d, dig[d]] = 1.0
            blk.attn.W_o.weight[HIST + d, HIST + d] = 1.0
        blk.attn.W_v.weight[C2, C1] = -1.0 / B
        blk.attn.W_o.weight[C2, C2] = 1.0
        for d in range(9):
            for h, off in ((2 * d, 0.0), (2 * d + 1, -1.0)):
                blk.mlp.fc1.weight[h, HIST:HIST + d + 1] = 5.0
                for p in range(5, 10):
                    blk.mlp.fc1.weight[h, POS + p - 5] = -(p - 5.0)
                blk.mlp.fc1.bias[h] = off
                sgn = 1.0 if off == 0.0 else -1.0
                blk.mlp.fc2.weight[LOG + d, h] += sgn
                if d + 1 < 9:
                    blk.mlp.fc2.weight[LOG + d + 1, h] -= sgn
        for d in range(9):
            model.head.weight[dig[d], LOG + d] = 1.0
            model.head.weight[dig[9], LOG + d] = -1.0
        model.head.weight[dig[9], C1] = 1.0 / B
