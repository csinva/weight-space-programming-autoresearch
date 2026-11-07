"""Snapshot: AntiIdentityReversal (acc 1.0, 5.04e3 params). Architecture: d_model = V+14 = 42, 1 layer, 1 head,
Block = x + attn(x) (no LayerNorm / MLP), final_ln = Identity, head.weight tied to token_emb.weight,
attention: W_q, W_k: 42->7, W_v: 42->28, W_o: 28->42, scores / sqrt(7), causal softmax. pos_emb has 22 rows.

write_weights (everything else zero; V=28, P=12, POS=V):
  token_emb[i, i] = 1
  key code (dims POS+0..6) on prompt positions t = 0..10:
      separator (t%4==3): dim 6 ;  letter: dim t//4 (word), dim 3 + t%4 (letter index)
  query code (dims POS+7..13) on answer position 11+j, j = 0..10 (its OWN word / letter):
      separator (j%4==3): dim 7+6 ;  letter: dim 7 + j//4 (word), dim 7+3 + j%4 (letter index)
  W_k[d, POS+d] = 1                      (d = 0..6)
  W_q[2-w, POS+7+w] = QG  (w = 0..2)     <- the word reversal: anti-identity on word dims
  W_q[d, POS+7+d]   = QG  (d = 3..6)     letter index and separator flag pass straight through
  W_v[i, i] = 1 ;  W_o[i, i] = OG        (attended token written into token dims; own token has value 1)
  QG = 10, OG = 2.   (QG=6 fails at 61%; QG>=7.5 works.)
"""
