import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Make each head have its own independent decay timescale in Layer 1
old_decay = """            # --- LAYER 1: Extremely sharp local extraction ---
            l1_decay = 15.0 + float(net) * (65.0 / 14.0) 
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay"""

new_decay = """            # --- LAYER 1: Independent Head Timescales ---
            # Instead of a single decay per network, we give each head in the network
            # a different decay rate, creating a dense spectrum of timescales
            base_decay = 10.0 + float(net) * (70.0 / 14.0)
            
            for h in range(10): # 10 heads per network (d_head=102, 1020 total / 10 = 102?)
                # Actually, there are 10 heads *total*, not per network.
                # n_heads = 10, d_model = 1020, d_head = 102
                # Wait, the h_start is (net % n_heads) * d_head. So each network only uses ONE head!
                pass
                
            l1_decay = 15.0 + float(net) * (65.0 / 14.0)
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay"""

content = content.replace(old_decay, new_decay)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Verify")

with open(filepath, "w") as f:
    f.write(content)
