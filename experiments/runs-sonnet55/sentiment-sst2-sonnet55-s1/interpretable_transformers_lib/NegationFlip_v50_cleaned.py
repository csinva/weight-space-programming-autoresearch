"""
Snapshot NegationFlip_v50_cleaned (sentiment-sst2, acc 0.785, 5.49e5 params).
Same file as interpretable_transformer.py at that point (run from the run folder, needs src/).

Hand-written circuit (no training, no LayerNorm), 3 layers:
  L1 attention: head k (k=1..N_SHIFT-1) attends to position p-k (sinusoidal position match) and copies
      that character's one-hot into slot k -> each position holds a window of its last N_SHIFT characters.
  L1 MLP: one ReLU neuron per lexicon stem (" good", " bad", ...). Fires iff the window ends with the stem.
      Writes +w / -w to SCORE.  Negator neurons (" not ", " n't", ...) write NEGF (flag) and CUE (-1).
  L2 attention: score(p, j) = cos(w(p-j)) + NEG_B*NEGF_j. Self wins unless a negator is within ~14 chars,
      so NEGNEAR = "negator shortly before me".  L2 MLP: 2 neurons flip SCORE's sign when NEGNEAR = 1.
  L3 attention: uniform average of SCORE + CUE over all positions -> AVG.
  Head: logit('1') = AVG + c1, logit('0') = -AVG + c0.

Key constants: N_SHIFT=7, WEAK_W=0.5, NEG_B=0.227, FLIP=2.0, CUE_W=1.0, PRIOR=0.006,
layers (heads, d_head, d_ff) = [(6, 33, len(lexicon)+8), (1, 3, 2), (1, 1, 0)], d_model = 7*33 + 8.
The lexicon is the POS/NEG/MORE_POS/MORE_NEG/WEAK_POS/WEAK_NEG/SUFFIXES/NEGATORS lists in
interpretable_transformer.py; copy that file for the complete word lists.
"""
