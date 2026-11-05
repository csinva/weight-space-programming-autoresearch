"""Snapshot: d43_B3_q16 (1 layer, 1 head, d_model=43, d_ff=20). Accuracy 1.0 on sort-five-digits.
Architecture defaults in SimpleTransformer for this snapshot: d_model=43, n_heads=1, n_layers=1, d_ff=20.
Only write_weights is shown; the rest of the file matches interpretable_transformer.py.
"""
import math
import torch


def write_weights(model, task) -> None:
    # Residual layout (d_model=43):
    #   0..9 digit one-hot | 10..20 position one-hot | 21,22 constants +B,-(B+2)
    #   23..32 digit histogram | 33..42 output dims (1 on the answer digit)
    # Constants keep the residual zero-mean so LayerNorm ~ constant rescale undone by its gain.
    # Attention: all positions attend uniformly to the 5 prompt digits -> digit counts.
    # MLP: at output step k, I_d = clamp(C_d - k, 0, 1) = [cumcount(<=d) > k]; logit_d = I_d - I_{d-1}.
    D = model.d_model
    B = 3.0
    TOK, POS, C1, C2, HIST, LOG = 0, 10, 21, 22, 23, 33
    for p in model.parameters():
        p.data.zero_()
    dig = [task.stoi[str(d)] for d in range(10)]
    with torch.no_grad():
        for d in range(10):
            model.token_emb.weight[dig[d], TOK + d] = 1.0
        model.token_emb.weight[:, C1] = B
        model.token_emb.weight[:, C2] = -(B + 1)
        model.token_emb.weight[dig, C2] = -(B + 2)
        for p in range(task.seq_len):
            model.pos_emb.weight[p, POS + p] = 1.0
        g = math.sqrt((2 + B * B + (B + 2) ** 2) / D)
        blk = model.blocks[0]
        for ln in (blk.ln1, blk.ln2, model.final_ln):
            ln.weight.fill_(g)
        blk.attn.W_q.weight[0, C1] = 16.0
        for p in range(5):
            blk.attn.W_k.weight[0, POS + p] = 1.0
        for d in range(10):
            blk.attn.W_v.weight[HIST + d, dig[d]] = 1.0
            blk.attn.W_o.weight[HIST + d, HIST + d] = 5.0
        blk.attn.W_v.weight[C2, C1] = -1.0 / B
        blk.attn.W_o.weight[C2, C2] = 5.0
        for d in range(10):
            for h, off in ((2 * d, 0.0), (2 * d + 1, -1.0)):
                blk.mlp.fc1.weight[h, HIST:HIST + d + 1] = 1.0
                for p in range(5, 10):
                    blk.mlp.fc1.weight[h, POS + p] = -(p - 5.0)
                blk.mlp.fc1.bias[h] = off
                sgn = 1.0 if off == 0.0 else -1.0
                blk.mlp.fc2.weight[LOG + d, h] += sgn
                if d + 1 < 10:
                    blk.mlp.fc2.weight[LOG + d + 1, h] -= sgn
        for d in range(10):
            model.head.weight[dig[d], LOG + d] = 10.0
