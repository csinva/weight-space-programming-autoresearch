# Snapshot: triangle_wave_d6 (acc 1.0, 460 params). d_model=6, 1 head, 1 layer, d_ff=10.
# Same as uniform_attn_triangle_wave but layer_norm sizes are (6,) and the parity output
# lives in residual dim 5 (fc2 row 5, fc2.bias[5] = -0.5, head weight[:, 5] = +1 for '1', -1 for '0').
