# Snapshot: triangle_wave_d2_fc2_scalar (acc 1.0, 47 params). Same as triangle_wave_d2_fc1_scalar but
# MLP.fc2 = Linear(d_ff, 1, bias=False) and forward returns F.pad(fc2(relu(fc1(x[..., :1]))), (1, 0)); fc2.weight[0, j-1] = slope change at k=j.
