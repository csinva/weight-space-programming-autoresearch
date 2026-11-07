import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Currently each network writes its positional decay to head+0.
# We will use head+0, head+1, and head+2 for fast/med/slow decays within the same network.
old_decay = """            # --- LAYER 1: Extremely sharp local extraction ---
            l1_decay = 15.0 + float(net) * (65.0 / 14.0) 
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75"""

new_decay = """            # --- LAYER 1: Multi-timescale Heads ---
            l1_decay_fast = 40.0 + float(net) * (60.0 / 14.0)
            l1_decay_med = 15.0 + float(net) * (65.0 / 14.0) 
            l1_decay_slow = 2.0 + float(net) * (20.0 / 14.0)
            
            # Fast Head
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay_fast
            
            # Medium Head
            l1_attn.W_q.weight[h_start + 1, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 1, d_start + 28] = l1_decay_med
            
            # Slow Head
            l1_attn.W_q.weight[h_start + 2, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 2, d_start + 28] = l1_decay_slow
            
            # Distribute the 28 features across the 3 timescales
            for i in range(9):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75
                
            for i in range(9, 18):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75
                
            for i in range(18, 28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75"""

content = content.replace(old_decay, new_decay)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Multi_Timescale_Heads")

with open(filepath, "w") as f:
    f.write(content)
print("Applied multi-timescale patch")
