# Snapshot: triangle_wave_d2_bias_only_fc1 (acc 1.0, 29 params). Same as triangle_wave_d2_scalar_emb_headbias but
# MLP has b1 (d_ff,) param only: hidden = relu(x[...,:1] + b1), b1[j-1] = -j for j=1..9; fc2 = Linear(9,1,bias=False) padded to dim 1.
