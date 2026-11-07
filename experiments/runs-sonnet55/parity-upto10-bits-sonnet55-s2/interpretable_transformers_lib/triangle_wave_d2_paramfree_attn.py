# Snapshot: triangle_wave_d2_paramfree_attn (acc 1.0, ~25 params). Same as triangle_wave_d2_bias_only_fc1 but
# CausalSelfAttention has no parameters: out = pad(cumsum(x[...,:1])/arange(1..T), to d_model) (causal mean of dim 0 -> dim 0).
