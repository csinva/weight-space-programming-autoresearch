# Snapshot: uniform_attn_triangle_wave (acc 1.0, 674 params). d_model=8, 1 head, 1 layer, d_ff=10.
# Same as uniform_attn_count_ramp_parity except the MLP block of write_weights is replaced by:
#
#        slopes = [((k + 1) % 2 - k % 2) / (us[k + 1] - us[k]) for k in range(10)]
#        for j in range(10):
#            blk.mlp.fc1.weight[j, 4] = 1.0
#            blk.mlp.fc1.bias[j] = -us[j]
#            blk.mlp.fc2.weight[6, j] = slopes[j] - (slopes[j - 1] if j else 0.0)
#        blk.mlp.fc2.bias[6] = -0.5
