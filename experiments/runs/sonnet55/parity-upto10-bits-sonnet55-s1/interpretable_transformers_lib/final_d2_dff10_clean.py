"""Snapshot: final_d2_dff10_clean (accuracy 1.0000, 64 params).

Model-specific parts only; the evaluation harness at the bottom of
interpretable_transformer.py is unchanged.

Architecture changes vs the template (in SimpleTransformer):
  * no position embedding (parity is order-free)
  * LayerNorms replaced by nn.Identity (ln1, ln2, final_ln)
  * CausalSelfAttention has no W_q/W_k: scores = zeros -> uniform causal attention
  * MLP Linear layers have bias=False
  * build_model: d_model=2, n_heads=1, n_layers=1, d_ff=10
"""

def write_weights(model, task) -> None:
    """Parity of the number of '1's in the prompt, exact, 2-dim residual stream, 64 parameters.

    Residual dims: dim 0 = "is a 1" -> later the count c -> later a = 2*parity(c);
                   dim 1 = "is '='" (the constant 1 at the answer position).
    '0' and '_' embed to the zero vector.  No LayerNorm, positions, biases, or q/k.

    1. Attention (uniform, since q = k = 0): at the final '=' the mean of the 11 prompt
       tokens' dim 0 is c/11; W_v scales by 11, so dim 0 becomes the integer count c.
    2. MLP, unit k = relu(c - k*[=]) = relu(c - k) for k = 0..9.  Writing
       a(c) = 2 relu(c) - 4 relu(c-1) + 4 relu(c-2) - ... is the zig-zag through
       (c, 2*(c mod 2)).  Unit 0 equals c, so its output weight is 2 - 1 to also cancel c.
    3. Head: logit('1') = a - [=] = +/-1, logit('0') = [=] - a.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        one, zero, eq = task.stoi["1"], task.stoi["0"], task.stoi["="]
        F, E = 0, 1
        n = task.prompt_len
        model.token_emb.weight[one, F] = 1.0
        model.token_emb.weight[eq, E] = 1.0
        blk = model.blocks[0]
        blk.attn.W_v.weight[F, F] = float(n)
        blk.attn.W_o.weight[F, F] = 1.0
        for k in range(n - 1):
            blk.mlp.fc1.weight[k, F] = 1.0
            blk.mlp.fc1.weight[k, E] = -float(k)
            blk.mlp.fc2.weight[F, k] = 1.0 if k == 0 else (-4.0 if k % 2 else 4.0)
        model.head.weight[one, F] = 1.0
        model.head.weight[one, E] = -1.0
        model.head.weight[zero, F] = -1.0
        model.head.weight[zero, E] = 1.0
