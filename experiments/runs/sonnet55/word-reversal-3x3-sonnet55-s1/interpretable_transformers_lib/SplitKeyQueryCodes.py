"""Snapshot: SplitKeyQueryCodes (acc 1.0). Diff vs SharedTokDims:
   d_model = V + 22 = 50. pos_emb[t] = e_{POS+t} for prompt positions t<11 (key code);
   pos_emb[11+j] = e_{POS+11+j} for answer steps j<11 (query code).
   W_k identity on key code; W_q[POS+SRC[j], POS+11+j] = 1 with SRC=[8,9,10,7,4,5,6,3,0,1,2].
   W_v/W_o copy token dims (gain 3) back into token dims; head reads token dims. 1 layer, 1 head, MLP zero.
"""
