# Snapshot LexiconShiftWindow_v1 (acc 0.665, 6.88e5 params).
# Layer-1: 8 heads (7 used) shift-copy previous chars into a window (sinusoidal-position attention, d_head=32, no LayerNorm).
# Layer-1 MLP: one neuron per lexicon stem " stem" (<=7 letters) ending at current char; writes +-1 to SCORE dim.
# Layer-2: uniform attention averages SCORE into AVG; head: '1' = AVG + 0.003*CONST, '0' = -AVG.
# Full source is the interpretable_transformer.py of that commit; lexicon: ~140 positive / ~150 negative stems.
# (See later snapshots for complete, runnable files.)
