# Snapshot: scalar_residual_d1 (acc 1.0, 24 params = tok 4 + b1 9 + fc2 9 + head 2). d_model=1, 1 layer, d_ff=9, no LayerNorm, no pos emb.
# token_emb = Embedding(vocab,1): E['1']=11, others 0. Attention: param-free causal mean -> x = k at '='.
# MLP: x += sum_{j=1..9} c_j ReLU(x - j), c_j = slope[j]-slope[j-1] (alternating +-2 pattern), b1[j-1]=-j, fc2 = Linear(9,1,no bias).
# After MLP x = triangle wave through (k, k mod 2). head = Linear(1,1) with weight 1, bias -0.5; logits = head(x) * sign
# (sign buffer: +1 for '1', -1 for '0', 0 else).
