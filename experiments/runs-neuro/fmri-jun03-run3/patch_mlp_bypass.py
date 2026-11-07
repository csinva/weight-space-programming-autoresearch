import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Wait! Is it possible that the MLPs are actually HARMING the exact integration?
# What if we set L1 MLP variance and L2 MLP variance to exactly 0?
# Then the network becomes a pure linear accumulator of exact word representations.
# Let's try pure linear accumulation!

replacement = """            # --- LAYER 1: Extremely sharp local extraction ---
            l1_decay = 15.0 + float(net) * (65.0 / 14.0) 
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            std_dev = 0.0001
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            # --- LAYER 2: Staggered Integration ---
            l2_decay = 0.05 + float(net) * (11.95 / 14.0)
            
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay
            
            # Staggered logic: 3-way split
            net_b = (net + 6) % 15
            net_c = (net + 12) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            
            for i in range(22):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0
                
            std_dev2 = 0.0001
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)"""

content = re.sub(
    r'            # --- LAYER 1: Extremely sharp local extraction ---\n            l1_decay = 15\.0 \+ float\(net\) \* \(65\.0 \/ 14\.0\).*?nn\.init\.normal_\(mlp2\.fc2\.weight\[d_start:d_start\+dim_per_net, f_start:f_start\+ff_per_net\], std=std_dev2 \* S\)',
    replacement,
    content,
    flags=re.DOTALL
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_Zero_MLP")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "from a 3-way split (+0, +6, +12) with MLPs completely zeroed out to test pure linear exponential accumulation.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
