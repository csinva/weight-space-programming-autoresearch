"""
Snapshot SinkCount_final2 -- digit-counting-10 (accuracy 1.0 on the 200-example eval).
Condensed copy: write_weights + constants only. The architecture classes (CausalSelfAttention,
MLP, Block with optional MLP, SimpleTransformer with n_heads=1, n_layers=2, d_ff=(0, 20), d_model=64)
and the evaluation harness are identical to SinkCount_final.py / interpretable_transformer.py.

Task: prompt = 11 digits + '=', answer = 2-digit count of the OTHER digits equal to the last digit.
Differences from SinkCount_final: no '=' penalty in the head (HEADPEN=0), constants are named.
"""
import torch
import torch.nn as nn

SQ, V_POS, GGAIN, A = 0.37, 4.3243, 0.1, 300.0
L1S, GATE_DIV, NC, EXCL, HEADPEN = 4.0, 30.0, 10, 3.0, 0.0
# layer-normed sink-mass reading for count m: h(m) = BETA + GAMMA/(m + R) (fitted; matches m=1..6 exactly)
BETA, GAMMA, R = -0.28503, 1.107, 6.158
THETA = [BETA + GAMMA / (u + 1 + R) for u in range(NC)]


def write_weights(model, task):
    # Residual layout: 0-9 token digit | 10 '=' | 11-26 position one-hot
    #                  27-36 copy of LAST digit | 37 sink mass g | 38-47 logit for digit u
    for p in model.parameters():
        p.data.zero_()
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):
            mod.weight.data.fill_(1.0)
    D = 10
    TOK, POS, COPY, G, OUT = 0, 11, 27, 37, 38
    with torch.no_grad():
        model.token_emb.weight[torch.arange(11), torch.arange(11)] = 1.0
        model.pos_emb.weight[torch.arange(16), POS + torch.arange(16)] = 1.0
        a1 = model.blocks[0].attn      # L1: copy the last digit to the answer positions
        for p in range(16):
            a1.W_q.weight[p if p <= 10 else 10, POS + p] = L1S
            a1.W_k.weight[p, POS + p] = 1.0
        for j in range(D):
            a1.W_v.weight[j, TOK + j] = 1.0
            a1.W_o.weight[COPY + j, j] = 0.2
        a2 = model.blocks[1].attn      # L2: sink mass g = 1/(c + m(E-1)) encodes the match count m
        for j in range(D):
            a2.W_q.weight[j, COPY + j] = SQ
            a2.W_k.weight[j, TOK + j] = 1.0
        a2.W_q.weight[10, POS + 11] = EXCL
        a2.W_q.weight[10, POS + 12] = EXCL
        a2.W_k.weight[10, POS + 12] = -EXCL      # mask pos-12's own key
        a2.W_v.weight[0, POS + 11] = 1.0         # '=' is the sink
        a2.W_o.weight[G, 0] = GGAIN
        mlp = model.blocks[1].mlp      # nearest-neighbour decode: relu pair per class gives -|h - theta_u|
        for u in range(NC):
            mlp.fc1.weight[2 * u, G], mlp.fc1.bias[2 * u] = A, -A * THETA[u]
            mlp.fc1.weight[2 * u + 1, G], mlp.fc1.bias[2 * u + 1] = -A, A * THETA[u]
            for k in (2 * u, 2 * u + 1):
                mlp.fc1.weight[k, POS + 12] = A / GATE_DIV     # gate: off at pos 11, ~0 at pos 12
                mlp.fc1.bias[k] -= (A / GATE_DIV) * V_POS
                mlp.fc2.weight[OUT + u, k] = -1.0
        for u in range(NC):
            model.head.weight[u, OUT + u] = 1.0
        model.head.weight[0, 10] = 1.0               # pos 11 ('='): tens digit is 0
        model.head.weight[10, POS + 12] = -HEADPEN


model_shorthand_name = "SinkCount_final2"
