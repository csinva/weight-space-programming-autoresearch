import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Currently L1 is FAST decay (15.0 to 80.0)
# Currently L2 is SLOW decay (0.05 to 12.0)
# What if we swap them? L1 does slow integration, L2 does fast extraction?
# In traditional transformers, lower layers capture local context, and higher layers capture long context.
# But here, we are doing Ridge Regression on the LAST layer output.
# So maybe the long context needs to be established first, and then the sharp token information added on top?

replacement = """            # --- LAYER 1: Extremely sharp local extraction ---
            # Wait, SWAPPING. Layer 1 is now SLOW decay!
            l1_decay = 0.05 + float(net) * (11.95 / 14.0)
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            std_dev = 0.01 + float(net) * (3.99 / 14.0)
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            # --- LAYER 2: Staggered Integration ---
            # Layer 2 is now FAST decay!
            l2_decay = 15.0 + float(net) * (65.0 / 14.0)
            
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
                
            std_dev2 = 0.5
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)"""

content = re.sub(
    r'            # --- LAYER 1: Extremely sharp local extraction ---\n            l1_decay = 15\.0 \+ float\(net\) \* \(65\.0 \/ 14\.0\).*?nn\.init\.normal_\(mlp2\.fc2\.weight\[d_start:d_start\+dim_per_net, f_start:f_start\+ff_per_net\], std=std_dev2 \* S\)',
    replacement,
    content,
    flags=re.DOTALL
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_Reversed_Layers")
content = content.replace("with L1 decay scale set to 15-80 instead of 10-80.", "with L1 doing slow integration and L2 doing fast extraction.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
