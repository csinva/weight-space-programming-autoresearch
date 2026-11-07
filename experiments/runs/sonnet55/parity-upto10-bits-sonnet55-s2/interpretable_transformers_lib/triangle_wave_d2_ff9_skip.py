# Snapshot: triangle_wave_d2_ff9_skip (acc 1.0, 67 params). d_model=2, d_ff=9.
# write_weights: E['1']=(11,0); W_v[0,0]=W_o[0,0]=1 (dim0 at '=' = k); us=k; slopes[k]=+-1.
#   for j=1..9: fc1.w[j-1,0]=1, fc1.b[j-1]=-j, fc2.w[1,j-1]=slopes[j]-slopes[j-1]; fc2.b[1]=-0.5
#   head['1']=(slopes[0], 1), head['0']=(-slopes[0], -1)   (head reads residual dim0=k for the first slope)
