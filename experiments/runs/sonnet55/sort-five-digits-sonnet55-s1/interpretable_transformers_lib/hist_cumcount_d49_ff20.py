"""Snapshot: hist_cumcount_d49_ff20 (1 layer, 1 head, d_model=49, d_ff=20). Accuracy 1.0 on sort-five-digits.

Sort digits: attention builds a digit histogram of the prompt; the MLP thresholds
cumulative counts against the output index k to emit the k-th smallest digit.
Defaults in SimpleTransformer for this snapshot: d_model=49, n_heads=1, n_layers=1, d_ff=20.
The architecture/harness code is identical to interpretable_transformer.py; only write_weights is shown.
"""
import math
import torch


def write_weights(model, task) -> None:
    # Residual layout (d_model=49):
    #   0..10 token one-hot | 11..26 position one-hot | 27,28 constants +B,-(B+2)
    #   29..38 digit histogram | 39..48 output dims (1 on the answer digit)
    D = model.d_model
    B = 30.0
    TOK, POS, C1, C2, HIST, LOG = 0, 11, 27, 28, 29, 39
    for p in model.parameters():
        p.data.zero_()
    dig = [task.stoi[str(d)] for d in range(10)]
    with torch.no_grad():
        for t in range(model.vocab_size):
            model.token_emb.weight[t, TOK + t] = 1.0
        model.token_emb.weight[:, C1] = B
        model.token_emb.weight[:, C2] = -(B + 2)
        for p in range(model.max_seq_len):
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
