"""Snapshot: uniform_attn_count_ramp_parity (accuracy 1.0, 844 params).

Full file = interpretable_transformer.py with the architecture below (1 layer, 1 head,
d_model=8, d_ff=20) and the write_weights() that follows. Architecture classes are the
unchanged defaults (CausalSelfAttention, MLP, Block, SimpleTransformer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(task, SimpleTransformer):
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len,
                              d_model=8, n_heads=1, n_layers=1, d_ff=20)
    write_weights(model, task)
    model.eval()
    return model


def write_weights(model, task) -> None:
    """Parity = (number of 1s) mod 2, read out at the '=' position.

    Attention (q=k=0 -> uniform over the 11 prompt tokens) averages a per-token vote
    (+1/2 for '1', -1/2 for '0' / pad / '=') into dim 4: a monotone function of k=#ones.
    The MLP computes parity(k) as an alternating sum of step functions [k>=j], each made of
    two ReLUs (a ramp). Dim 6 holds parity-0.5; the head reads +-dim 6.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        tok = task.stoi
        E = model.token_emb.weight
        E[tok['1'], 0], E[tok['1'], 1] = 1, -1
        E[tok['0'], 0], E[tok['0'], 1] = -1, 1
        for t in ('_', '='):
            E[tok[t], 2], E[tok[t], 3] = 1, -1
        s = F.layer_norm(E[tok['1']], (8,))[0].item()
        blk = model.blocks[0]
        blk.attn.W_v.weight[4, 0] = 1 / (2 * s)
        blk.attn.W_v.weight[4, 2] = -1 / (2 * s)
        blk.attn.W_o.weight[4, 4] = 1.0
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
        us = []
        for k in range(11):
            x = E[tok['=']].clone()
            x[4] = (2 * k - 11) / 22
            us.append(F.layer_norm(x, (8,))[4].item())
        for j in range(1, 11):
            w = 1.0 / (us[j] - us[j - 1])
            sign = 1.0 if j % 2 == 1 else -1.0
            a, b = 2 * (j - 1), 2 * (j - 1) + 1
            blk.mlp.fc1.weight[a, 4] = w; blk.mlp.fc1.bias[a] = -w * us[j - 1]
            blk.mlp.fc1.weight[b, 4] = w; blk.mlp.fc1.bias[b] = -w * us[j]
            blk.mlp.fc2.weight[6, a] = sign
            blk.mlp.fc2.weight[6, b] = -sign
        blk.mlp.fc2.bias[6] = -0.5
        model.head.weight[tok['1'], 6] = 1.0
        model.head.weight[tok['0'], 6] = -1.0


model_shorthand_name = "uniform_attn_count_ramp_parity"
