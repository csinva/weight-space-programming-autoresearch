"""
Snapshot clean_final_d56 (accuracy 1.0 on 200 and on 1000 samples, 1.75e4 params).
Task: linear-interpolation-two-points. Prompt "x1,yy;x2,yy;q=" -> y1 + s*(q-x1), s=(y2-y1)/(x2-x1) in 1..5.
Positions 11 ('=') and 12 (tens digit) emit the two answer digits.

Architecture: no LayerNorm; attention only in block 0 (7 heads, d_head=2); d_model=56; 4 blocks;
MLP widths (15, 11, 18, 60).  Residual layout:
  TOK 0..12 token one-hot | POS 13..26 position one-hot | SLOT 27..33 (7 gathered digit values)
  SD 34..38 slope one-hot | A 39 answer scalar | TENS 40 tens scalar | OUT 41..50 digit logits
SRC_POS = [0, 2, 3, 5, 7, 8, 10]  -> x1, y1 tens, y1 ones, x2, y2 tens, y2 ones, q

Weights (all closed form, see write_weights in interpretable_transformer.py of this run):
 B0 attn : head h: q = 30*sqrt(dh)*[pos in {11,12}], k = [pos == SRC_POS[h]], v = numeric digit value -> SLOT_h.
 B0 MLP  : SD[s-1] = tri(dy - s*dx), tri(L) = relu(L+1) - 2 relu(L) + relu(L-1)  (1 iff L == 0).
 B1 MLP  : A = y1 + sum_s +-relu(+-s*(q-x1) + 50*(SD[s-1]-1))  (gated product).
 B2 MLP  : TENS = sum_{k=1..9} [A >= 10k]   (step = relu(A-10k+1) - relu(A-10k)).
 B3 MLP  : OUT[k] = tri_gated(A - 10*TENS - k, pos 12) + tri_gated(TENS - k, pos 11); head reads OUT.

NOTE: this file is a description snapshot; the exact code is the final interpretable_transformer.py in the run folder.
"""
