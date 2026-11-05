# Snapshot: triangle_wave_d2_nolayernorm (acc 1.0, 96 params). d_model=2, 1 layer, no LayerNorm (Identity),
# attention = causal mean of scalar value channel, d_ff=10, pos_emb zero (12x2), max_seq_len=task.seq_len.
# write_weights:
#   zero all; E['1']=(1,0); W_v[0,0]=1; W_o[0,0]=1  -> dim0 at '=' is k/11
#   us=[k/11 for k in 0..10]; slopes[k]=((k+1)%2-k%2)/(us[k+1]-us[k]) for k<10
#   fc1.w[j,0]=1, fc1.b[j]=-us[j]; fc2.w[1,j]=slopes[j]-slopes[j-1]; fc2.b[1]=-0.5
#   head['1',1]=+1, head['0',1]=-1
