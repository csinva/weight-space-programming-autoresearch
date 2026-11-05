# Snapshot: scalar_residual_d1_verified (200/200 eval, 1024/1024 exhaustive full-length prompts, 28 params).
# Also covers scalar_residual_d1_final (identical weights, no self-check). Architecture: d_model=1, 1 layer, d_ff=9,
# no LayerNorm (Identity), no positional embedding, softmax attention with 1x1 W_q=W_k=0, W_v=W_o=1,
# MLP = bias-only knots b1[j-1] = -j (j=1..9) + fc2 = Linear(9,1,bias=False), head = Linear(1,1) with sign buffer.
#
# write_weights:
#   zero all params
#   attn.W_v = attn.W_o = 1;  token_emb['1'] = 11 (others 0)   -> x = k (number of ones) at '='
#   us = [0..10]; slopes[k] = ((k+1)%2 - k%2) / (us[k+1]-us[k])      # +1,-1,+1,...
#   for j in 1..9: mlp.b1[j-1] = -j; mlp.fc2.weight[0, j-1] = slopes[j] - slopes[j-1]   # -2,+2,-2,...
#   head.bias = -0.5; head.weight = 1; sign['1'] = +1, sign['0'] = -1
# Forward: x = emb; x += attn(x); x += fc2(relu(x + b1)); logits = (head(x)) * sign   -> x = triangle wave of k.
