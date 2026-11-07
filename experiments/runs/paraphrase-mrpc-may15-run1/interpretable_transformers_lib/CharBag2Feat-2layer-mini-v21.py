"""iter21 snapshot: 2-layer minimal d_model=36 variant.

Compact 2-layer design that fits the char-bag diff into the SAME channels as
the token embedding (chans 0..32). Layer 0 attends to s1 and writes +avg_s1
into chans 0..32 at the '=' position; layer 1 attends to s2 and writes
-avg_s2 into the same chans (giving diff). Layer 1 MLP then computes the
two-feature (weighted_L1 + len-diff) score.

Architecture (d_model=36, n_heads=1, n_layers=2, d_ff=68):
  chans 0..32 : token char one-hot at non-eq positions, becomes diff-bag at eq.
  chan  33    : is_s1 indicator.
  chan  34    : is_s2 indicator.
  chan  35    : is_eq indicator (Q-routing channel at pos 121).

Result: 0.7050 acc with ~27k params (vs v17's 0.73 with 41k params). The
forced channel-reuse imposes more LN cross-talk than the v17 design and the
accuracy drops. Kept as documentation of an attempted compaction.
"""
# (See iter21 sweep script for full weight assignments; this file is a marker.)
