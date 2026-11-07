# Snapshot: triangle_wave_d2_scalar_head (acc 1.0, 41 params). Same as triangle_wave_d2_fc2_scalar but
# head = Linear(2,1,bias=False); logits = head(h) * sign, sign buffer = +1 on '1', -1 on '0', 0 elsewhere.
# head.weight = [slopes[0], 1]; E['1']=(11,0), E['=']=(0,-0.5).
