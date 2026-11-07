# Snapshot: triangle_wave_d3_meanattn (acc 1.0, 157 params). Same weights as triangle_wave_d3_seq12, but
# CausalSelfAttention is: v = W_v(x) (Linear 3->1); out = W_o(cumsum(v)/arange(1..T)) (Linear 1->3). No q/k.
