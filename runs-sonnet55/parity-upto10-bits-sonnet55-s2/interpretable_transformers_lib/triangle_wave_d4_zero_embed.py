# Snapshot: triangle_wave_d4_zero_embed (acc 1.0, 278 params). d_model=4, 1 head, 1 layer, d_ff=10.
# Same as triangle_wave_d4 except: E['0']=E['_']=0 vector, E['=']=(0,0,1,-1), E['1']=(1,-1,0,0);
# W_v has a single entry W_v[0,0]=1/(2s) (no pad vote); us[k] uses x[0] += k/22.
