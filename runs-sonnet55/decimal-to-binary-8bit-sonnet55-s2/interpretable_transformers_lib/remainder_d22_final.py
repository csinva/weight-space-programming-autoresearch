"""Snapshot: remainder_d22_final (model definition + write_weights only; the evaluation
harness is identical to interpretable_transformer.py).  Build with
    SimpleTransformer(vocab_size, max_seq_len=16, d_model=22, n_heads=11, n_layers=1, d_ff=8)
and Block(..., use_attn=True).  Same Block/MLP/CausalSelfAttention classes as the main file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def write_weights(model, task) -> None:
    """Long-division circuit.  Position 3+j predicts bit j (MSB first).

    Attention (10 heads, head p fixed on source position p) sums
        R = 100a + 10b + c - sum_{i<j} 2^(7-i) * bit_i      (scaled by eps)
    i.e. the remainder r_j of N after removing the already emitted bits.  Heads whose
    source bit has not been emitted yet attend to the '=' sink (position 3, value 0).
    MLP: unit j (gated to position 3+j) fires iff r_j >= 2^(7-j), writing the '1' logit;
    '0' wins by a constant default logit.
    Residual (d_model=22): 0-10 token one-hot | 11-21 position one-hot.
    """
    TOK, POS, R, LOGIT = 0, 11, 9, 2
    D = model.d_model
    dh = D // model.n_heads
    stoi = task.stoi
    eps, K, G = 0.31 / 100, 30.0, 4.0
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
        for t in range(task.vocab_size):
            model.token_emb.weight[t, TOK + t] = 1.0
        for p_ in range(11):
            model.pos_emb.weight[p_, POS + p_] = 1.0
        NONDIGIT = [stoi["0"], stoi["1"], stoi["="]]

        att = model.blocks[0].attn
        src_pos = [0, 1, 2] + [4 + i for i in range(7)]
        coef = [100, 10, 1] + [-(2 ** (7 - i)) for i in range(7)]
        for h, (sp, c) in enumerate(zip(src_pos, coef)):
            for t in NONDIGIT:
                att.W_q.weight[h * dh, TOK + t] = 1.0
            att.W_k.weight[h * dh, POS + sp] = K
            if sp > 3:
                att.W_k.weight[h * dh, POS + 3] = K / 4  # sink
            if sp < 3:
                for d in range(10):
                    att.W_v.weight[h * dh + 1, TOK + stoi[str(d)]] = eps * c * (d - 4.5)
            else:
                att.W_v.weight[h * dh + 1, TOK + stoi["1"]] = eps * c
            att.W_o.weight[R, h * dh + 1] = 1.0

        mlp = model.blocks[0].mlp
        for j in range(8):
            mlp.fc1.weight[j, R] = 1.0
            mlp.fc1.weight[j, POS + 3 + j] = G
            mlp.fc2.weight[LOGIT + 1, j] = 1.0

        def pre_at(n, j):
            s = "%03d=" % n + "0" * j
            ids = torch.tensor([[stoi[ch] for ch in s]])
            h = model.token_emb(ids) + model.pos_emb(torch.arange(ids.shape[1]))[None]
            h = h + att(model.blocks[0].ln1(h))
            return mlp.fc1.weight @ F.layer_norm(h, (D,))[0, -1]

        for j in range(8):
            hi, lo = pre_at(2 ** (7 - j), j), pre_at(2 ** (7 - j) - 1, j)
            mlp.fc1.bias[j] = -(hi[j] + lo[j]) / 2
            mlp.fc2.weight[:, j] /= (hi[j] - lo[j]) / 2

        model.head.weight[stoi["1"], LOGIT + 1] = 1.0
        for t in NONDIGIT:
            model.head.weight[stoi["0"], TOK + t] = 0.5


model_shorthand_name = "remainder_d22_final"
