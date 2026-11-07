# Snapshot: scalar_residual_d1_vote1 (acc 1.0, 24 params). Same as scalar_residual_d1 but E['1']=1 and the "attention"
# is a causal cumulative SUM (no division by position). Caveat: a sum is not a softmax attention; it only works because the prompt length is fixed.
