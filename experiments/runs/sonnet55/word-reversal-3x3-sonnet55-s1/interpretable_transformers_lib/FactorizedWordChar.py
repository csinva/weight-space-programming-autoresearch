"""Snapshot: FactorizedWordChar (acc 1.0, 1.05e4 params). Diff vs earlier snapshots:
   - Block has no MLP / ln2 (attention-only routing).
   - d_model = V + 14 = 42, n_heads=1, n_layers=1.
   - Position code is (word one-hot 3 | char one-hot 4):
       key code  dims POS+0..6 on prompt positions 0..10,
       query code dims POS+7..13 on answer positions 11+j, holding (word, char) of source SRC[j].
       W_k = identity on dims 0..6, W_q[POS+d, POS+7+d] = QG (4.0).
   - SRC = [8,9,10,7,4,5,6,3,0,1,2]; W_v / W_o copy token dims (gain OG=3) back to token dims; head reads token dims.
"""
