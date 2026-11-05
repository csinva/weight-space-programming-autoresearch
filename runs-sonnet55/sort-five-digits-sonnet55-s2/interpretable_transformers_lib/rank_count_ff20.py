# Snapshot: same as rank_count_v1 (d_model=44, 1 head, 1 layer) but d_ff=20 and the MLP section is:
#   specs = (("B", 0.0, 1.0), ("B", -1.0, -1.0))
#   for d in range(10):
#       for typ, delta, out_sign in specs:
#           out_sign = out_sign * (10 - d)   # smallest d with cum_le(d) > k wins
#           (fc1 row: pos weights -k/P on POS+5+k, +1 on BAG e<=d, const fixed by LN mean; fc2 -> LOG+d)
# Full file is the working interpretable_transformer.py at the time (acc 1.0, 200/200).
