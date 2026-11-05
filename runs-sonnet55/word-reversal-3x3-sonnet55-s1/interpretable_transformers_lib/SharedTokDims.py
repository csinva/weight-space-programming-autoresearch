"""Snapshot: SharedTokDims (acc 1.0). Diff vs OneHeadPosCopy:
   d_model = V + T (51); OUT = 0 (attention writes into token dims); W_o gain 3.0 so the attended token
   outweighs the residual's own-token one-hot. Everything else identical (1 layer, 1 head, d_ff=1, MLP zero).
"""
