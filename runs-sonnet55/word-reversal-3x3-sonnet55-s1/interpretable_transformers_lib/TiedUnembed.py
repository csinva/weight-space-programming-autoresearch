"""Snapshot: TiedUnembed (acc 1.0, 5.08e3 params). Architecture edits vs template:
  - CausalSelfAttention: single head, W_q/W_k: d_model->7, W_v: d_model->28, W_o: 28->d_model (no bias), scores/sqrt(7).
  - Block: x + attn(x)   (no LayerNorm, no MLP);  final_ln = Identity;  head.weight tied to token_emb.weight.
  - d_model = V + 14 = 42, n_layers = 1, n_heads = 1.
write_weights (all other params zero):
  token_emb[i, i] = 1
  SRC = [8,9,10,3,4,5,6,3,0,1,2]  # prompt position read at answer step j (words reversed; '_' from pos 3)
  prompt position t<11:   pos_emb[t, V + t//4] = pos_emb[t, V+3 + t%4] = 1        (key code: word, char)
  answer step j (pos 11+j): pos_emb[11+j, V+7 + SRC[j]//4] = pos_emb[11+j, V+10 + SRC[j]%4] = 1  (query code)
  W_k[d, V+d] = 1,  W_q[d, V+7+d] = QG (10),  d<7
  W_v[i, i] = 1,  W_o[i, i] = OG (2)  (writes attended token into the token dims, beating the own token's 1)
"""
