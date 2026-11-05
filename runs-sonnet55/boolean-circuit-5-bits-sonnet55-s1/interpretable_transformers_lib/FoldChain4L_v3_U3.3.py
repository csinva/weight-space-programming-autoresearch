"""
Interpretable transformer, task boolean-circuit-5-bits (snapshot FoldChain4L_v3_U3.3).

Prompt "b0 o1 b1 o2 b2 o3 b3 o4 b4 =" is evaluated strictly LEFT-TO-RIGHT
(no operator precedence), operators & | ^. Answer is one bit.

Circuit: 4 layers; layer l computes the running value v_l = o_l(v_{l-1}, b_l).
Layers 1-3 act at position 2l (the bit b_l); layer 4 acts at the '=' position.
Each layer: 3 attention heads fetch (operator, previous value, bit) into
dedicated residual slots, then a 12-neuron MLP is an AND-lookup over
(operator, bit, prev value, position) that writes the new value v_l.
Only the core pieces are reproduced here; see interpretable_transformer.py
for the full harness (unchanged below `write_weights`).
"""

import torch
import torch.nn as nn

SCALE_V = 0.17       # token one-hot after LN1 has amplitude ~5.75 -> copy with 1/5.75
SV_STATE = 0.127     # running-value dims (large, dominate the norm) -> LN1 gives ~7.9
UNIT = 3.3           # post-LN2 amplitude of an active feature


def write_weights(model, task) -> None:
    """Residual layout (d_model=64): tok one-hot 0-5 ('0','1','&','|','^','='),
    pos one-hot 6-16, then one 9-dim block per layer l=1..4:
      O_l (3: op &,|,^), B_l (2: bit), P_l (2: previous value), V_l (2: new value).
    Heads: 0 fetches the operator, 1 the previous value, 2 the bit.
    MLP_l: 12 neurons = AND(op, bit, prev, position) lookup -> writes V_l.
    Config: SimpleTransformer(n_layers=4, d_ff=12), d_model=64, 4 heads.
    """
    E, G, SV, U, M = 1.0, 40.0, SCALE_V, UNIT, 40.0
    H = 16
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
        for i in range(6):
            model.token_emb.weight[i, i] = E
        for p in range(11):
            model.pos_emb.weight[p, 6 + p] = E
        for l in range(1, 5):
            blk = model.blocks[l - 1]
            base = 17 + 9 * (l - 1)
            O, B, P, V = base, base + 3, base + 5, base + 7
            q = 9 if l == 4 else 2 * l
            keys = [7, 6, 8] if l == 4 else [q - 1, q - 2, q]
            src_prev = 0 if l == 1 else 17 + 9 * (l - 2) + 7
            at = blk.attn
            for h, kp in enumerate(keys):
                at.W_q.weight[H * h + kp, 6 + q] = G
                for j in range(11):
                    at.W_k.weight[H * h + j, 6 + j] = 1.0
            for j in range(3):
                at.W_v.weight[0 * H + j, 2 + j] = SV
                at.W_o.weight[O + j, 0 * H + j] = 1.0
            for j in range(2):
                at.W_v.weight[1 * H + j, src_prev + j] = SV if l == 1 else SV_STATE
                at.W_o.weight[P + j, 1 * H + j] = 1.0
                at.W_v.weight[2 * H + j, j] = SV
                at.W_o.weight[B + j, 2 * H + j] = 1.0
            n = 0
            for op in range(3):
                for b in range(2):
                    for v in range(2):
                        res = [b & v, b | v, b ^ v][op]
                        blk.mlp.fc1.weight[n, O + op] = 1.0
                        blk.mlp.fc1.weight[n, B + b] = 1.0
                        blk.mlp.fc1.weight[n, P + v] = 1.0
                        blk.mlp.fc1.weight[n, 6 + q] = 1.0
                        blk.mlp.fc1.bias[n] = -3.5 * U
                        blk.mlp.fc2.weight[V + res, n] = M / (0.5 * U)
                        n += 1
        model.head.weight[0, 17 + 27 + 7] = 1.0
        model.head.weight[1, 17 + 27 + 8] = 1.0


model_shorthand_name = "FoldChain4L_v3_U3.3"
model_description = "4-layer sequential left-to-right fold: layer l computes running value at pos 2l (last layer at '=') via 3 gather heads + 12-neuron AND lookup MLP."
