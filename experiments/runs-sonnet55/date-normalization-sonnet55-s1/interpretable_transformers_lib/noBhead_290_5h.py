"""noBhead_290_5h (200/200): diff vs ClosersRoles2L_compact288.py
  - alloc: C3 has 12 dims (dim 11 = '.' flag), no separate "B" slot
  - layer-1 head 4 (absolute-position read of char 4) removed; heads 0-3 unchanged, C3 head also writes '.' flag
  - B = n("C3")+11 is used by the two end-of-text units (year if not B, day if B)
  - SimpleTransformer defaults: d_model=290, n_heads=5
"""
