# Snapshot: triangle_wave_d4 (acc 1.0, 278 params). d_model=4, 1 head, 1 layer, d_ff=10.
# write_weights():
#   zero all params; LN weights = 1
#   E['1']=(1,-1,0,0), E['0']=(-1,1,0,0), E['_']=E['=']=(0,0,1,-1)
#   s = layer_norm(E['1'])[0];  W_v[0,0]=1/(2s), W_v[0,2]=-1/(2s);  W_o[0,0]=1   (q=k=0: uniform attention)
#   us[k] = layer_norm(E['='] + ((2k-11)/22) e_0)[0]  for k=0..10 ones
#   slopes[k] = ((k+1)%2 - k%2)/(us[k+1]-us[k]);  for j<10: fc1.w[j,0]=1, fc1.b[j]=-us[j],
#   fc2.w[1,j] = slopes[j]-slopes[j-1];  fc2.b[1] = -0.5;  head[1,'1']=+1, head[1,'0']=-1
