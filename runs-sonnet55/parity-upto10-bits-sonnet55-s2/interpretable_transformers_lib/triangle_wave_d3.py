# Snapshot: triangle_wave_d3 (acc 1.0, 199 params). d_model=3, 1 head, 1 layer, d_ff=10.
# Same as triangle_wave_d4_zero_embed except: E['1']=(1,-1,0), E['=']=(0,1,-1), '0'/'_' = 0;
# fc2.bias[1] = -2.5; head['1']=(0,1,-1), head['0']=(0,-1,1)  (zero-sum direction x1-x2, immune to LN mean).
