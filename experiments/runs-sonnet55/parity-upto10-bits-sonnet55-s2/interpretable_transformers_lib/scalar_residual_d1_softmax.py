# Snapshot: scalar_residual_d1_softmax (acc 1.0, 28 params). d_model=1; CausalSelfAttention is genuine softmax attention
# with 1x1 W_q=W_k=0 (uniform over the causal prefix), W_v=W_o=1; E['1']=11 so the attention mean at '=' is k.
# MLP (bias-only fc1, knots at j=1..9, fc2 = slope changes), head = x - 0.5 times sign buffer. Otherwise as scalar_residual_d1.
