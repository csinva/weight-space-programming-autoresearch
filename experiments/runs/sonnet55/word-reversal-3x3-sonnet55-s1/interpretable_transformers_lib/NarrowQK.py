"""Snapshot: NarrowQK (acc 1.0, 6.26e3 params). Same weights as NoLayerNorms_QG40 but attention is
narrow: W_q, W_k: d_model->7 (rows = word one-hot 3 | char one-hot 4), W_v: d_model->28, W_o: 28->d_model.
  W_k[d, POS+d] = 1;  W_q[d, POS+7+d] = QG (20);  W_v[i, i] = 1;  W_o[i, i] = OG (2);  head[i, i] = 1.
  SRC = [8,9,10,7,4,5,6,3,0,1,2]; pos_emb on prompt t<11: word t//4, char t%4; on answer step j: word/char of SRC[j].
  No LayerNorm / MLP; final_ln = Identity.
"""
