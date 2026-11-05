"""Snapshot: uniform_attn_count_relu_zigzag (accuracy 1.0000, 674 params).

Only the model-specific parts are reproduced here; the rest of
interpretable_transformer.py (architecture classes, evaluation harness) is unchanged.
build_model uses: d_model=8, n_heads=1, n_layers=1, d_ff=10.
"""

def write_weights(model, task) -> None:
    # Residual dims: 0..3 one-hot token, 4/5 = (+f,-f) count signal, 6/7 = (+d,-d) answer.
    # Attention (q=k=0 -> uniform over prefix) at the final '=' averages a '1'-indicator
    # over the 11 prompt chars -> f is affine in the count c.  The MLP has one ReLU per
    # knot, building the zig-zag y(c)=+1 odd / -1 even (knots computed through LayerNorm).
    # Head: logit('1') = +d, logit('0') = -d.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)
        D = model.d_model
        one, zero, eq = task.stoi["1"], task.stoi["0"], task.stoi["="]
        for t in range(task.vocab_size):
            model.token_emb.weight[t, t] = 1.0
        blk = model.blocks[0]
        sig_in = math.sqrt(1 / D - 1 / D ** 2)
        blk.attn.W_v.weight[4, one] = 1.0
        blk.attn.W_v.weight[5, one] = -1.0
        blk.attn.W_o.weight.copy_(torch.eye(D))

        def ln(x, eps=1e-5):
            x = x - x.mean()
            return x / torch.sqrt((x ** 2).mean() + eps)

        n = task.prompt_len
        zs = []
        for c in range(n):
            f = (c / n - 1 / D) / sig_in
            x = torch.zeros(D)
            x[eq] = 1.0
            x[4] = f
            x[5] = -f
            u = ln(x)
            zs.append((u[4] - u[5]).item())
        y = [1.0 if c % 2 else -1.0 for c in range(n)]
        prev = 0.0
        for k in range(n - 1):
            slope = (y[k + 1] - y[k]) / (zs[k + 1] - zs[k])
            blk.mlp.fc1.weight[k, 4] = 1.0
            blk.mlp.fc1.weight[k, 5] = -1.0
            blk.mlp.fc1.bias[k] = -zs[k]
            blk.mlp.fc2.weight[6, k] = slope - prev
            blk.mlp.fc2.weight[7, k] = -(slope - prev)
            prev = slope
        blk.mlp.fc2.bias[6] = y[0]
        blk.mlp.fc2.bias[7] = -y[0]
        model.head.weight[one, 6] = 1.0
        model.head.weight[one, 7] = -1.0
        model.head.weight[zero, 6] = -1.0
        model.head.weight[zero, 7] = 1.0
