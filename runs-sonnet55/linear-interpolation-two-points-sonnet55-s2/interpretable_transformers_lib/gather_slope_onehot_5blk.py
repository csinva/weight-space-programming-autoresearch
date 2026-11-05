"""
Snapshot gather_slope_onehot_5blk (accuracy 1.0, 5 blocks, d_model=144, no LayerNorm, attn only in block 0).
Task: linear-interpolation-two-points. Prompt "x1,yy;x2,yy;q=" -> y1 + s*(q-x1), s=(y2-y1)/(x2-x1) in 1..5.

Circuit:
 B0 attn: 7 heads, head h attends (from positions 11/12) to fixed source position SRC_POS[h], copies the digit one-hot into SLOT_h.
 B0 MLP : slope one-hot SD[s] = [dy - s*dx == 0] using triangle of 3 ReLUs on linear scalars.
 B1 MLP : A = y1 + sum_s SD[s]*s*(q-x1) via gated products relu(+-s*t + M*(SD[s]-1)).
 B2 MLP : TENS one-hot from scalar A via trapezoids.   B3 MLP: ONES one-hot from A-10*tens.
 B4 MLP : position-gated readout (pos 11 -> tens, pos 12 -> ones); head reads OUT.
See interpretable_transformer.py history for the full code (identical weights construction).
"""
