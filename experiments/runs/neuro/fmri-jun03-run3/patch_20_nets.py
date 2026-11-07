import os
import math
import torch
import torch.nn as nn

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Replace num_nets and dim_per_net
old_params = """        num_nets = 15
        dim_per_net = 64
        ff_per_net = 256"""

new_params = """        num_nets = 20
        dim_per_net = 48
        ff_per_net = 192"""

# Update L2 staggered logic
old_staggered = """            # Staggered logic: 3-way split
            net_b = (net + 6) % 15
            net_c = (net + 12) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            
            for i in range(22):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.15
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 0.85"""

new_staggered = """            # Staggered logic: 3-way split for 20 nets
            net_b = (net + 7) % 20
            net_c = (net + 14) % 20
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            
            for i in range(16):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.15
                
            for i in range(16):
                l2_attn.W_v.weight[h_start + 16 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 16 + i, h_start + 16 + i] = S * 1.0
                
            for i in range(16):
                l2_attn.W_v.weight[h_start + 32 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 32 + i, h_start + 32 + i] = S * 0.85"""

old_l1_decay = """            l1_decay = 15.0 + float(net) * (65.0 / 14.0)"""
new_l1_decay = """            l1_decay = 15.0 + float(net) * (65.0 / 19.0)"""

old_l2_decay = """            l2_decay = 0.01 + float(net) * (13.99 / 14.0)"""
new_l2_decay = """            l2_decay = 0.01 + float(net) * (13.99 / 19.0)"""

old_std_dev2 = """            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)"""
new_std_dev2 = """            std_dev2 = 0.01 + float(net) * (3.99 / 19.0)"""

content = content.replace(old_params, new_params)
content = content.replace(old_staggered, new_staggered)
content = content.replace(old_l1_decay, new_l1_decay)
content = content.replace(old_l2_decay, new_l2_decay)
content = content.replace(old_std_dev2, new_std_dev2)

content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_20_Nets")

with open(filepath, "w") as f:
    f.write(content)
print("Applied 20 nets patch")
