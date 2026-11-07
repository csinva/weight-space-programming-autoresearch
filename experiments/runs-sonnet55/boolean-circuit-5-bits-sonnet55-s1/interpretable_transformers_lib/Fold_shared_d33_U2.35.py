# Snapshot of write_weights() + config for Fold_shared_d33_U2.35 (accuracy 1.0, d_model=33, 3 heads, 4 layers, d_ff=12).
# Harness (Block/Attention/MLP/eval main) identical to interpretable_transformer.py's unmodified parts.
# build_model: SimpleTransformer(vocab, max_seq_len=16, n_layers=4, d_ff=12, d_model=33, n_heads=3)

def write_weights(model, task) -> None:
    """Left-to-right fold v_l = op_l(v_{l-1}, b_l) over "b0 o1 b1 o2 b2 ... o4 b4 =".
    Residual (33 dims): tok one-hot 0-5, pos one-hot 6-16, shared slots O(17-19) op, B(20-21) bit,
    P(22-23) previous value, then V_l (2 dims each, 24+2(l-1)) per layer.
    Layer l<=3 acts at pos 2l; layer 4 at '=' (pos 9). Heads: 0 op, 1 prev value, 2 bit.
    MLP_l: 12 neurons = AND(op, bit, prev) lookup -> writes V_l (large magnitude so LN normalises it)."""
    E, G, SV, U, M = 1.0, 40.0, SCALE_V, UNIT, 40.0
    H = model.d_model // model.n_heads
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
            O, B, P, V = 17, 20, 22, 24 + 2 * (l - 1)
            q = 9 if l == 4 else 2 * l
            keys = [7, 6, 8] if l == 4 else [q - 1, q - 2, q]
            src_prev = 0 if l == 1 else 24 + 2 * (l - 2)
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
                        blk.mlp.fc1.bias[n] = -2.5 * U
                        blk.mlp.fc2.weight[V + res, n] = M / (0.5 * U)
                        n += 1
        model.head.weight[0, 30] = 1.0
        model.head.weight[1, 31] = 1.0

SCALE_V = 0.242
SV_STATE = 0.176
UNIT = 2.35
