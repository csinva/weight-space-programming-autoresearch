import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

with open(filepath, "r") as f:
    content = f.read()

# Change n_layers from 2 to 3
content = content.replace("n_layers: int = 2", "n_layers: int = 3")

# We need to define l3_attn and mlp3
# Find:
#         l2_attn = model.blocks[1].attn
#         mlp2 = model.blocks[1].mlp
#         
#         for net in range(num_nets):

l3_defs = """        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        l3_attn = model.blocks[2].attn
        mlp3 = model.blocks[2].mlp
        
        for net in range(num_nets):"""

content = content.replace(
    "        l2_attn = model.blocks[1].attn\n        mlp2 = model.blocks[1].mlp\n        \n        for net in range(num_nets):",
    l3_defs
)

# Now we need to add L3 logic after L2 logic
# Find:
#             nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
#             nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)

l3_logic = """            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)
            
            # --- LAYER 3: Extremely Slow Decay Integration ---
            l3_decay = 0.01 + float(net) * (2.0 / 14.0)
            
            l3_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l3_attn.W_k.weight[h_start + 0, d_start + 28] = l3_decay
            
            net_d = (net + 3) % 15
            net_e = (net + 9) % 15
            d_start_d = net_d * dim_per_net
            d_start_e = net_e * dim_per_net
            
            for i in range(22):
                l3_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l3_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(21):
                l3_attn.W_v.weight[h_start + 22 + i, d_start_d + i] = 1.0
                l3_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l3_attn.W_v.weight[h_start + 43 + i, d_start_e + i] = 1.0
                l3_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0
                
            std_dev3 = std_dev2 * 0.5
            nn.init.normal_(mlp3.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev3)
            nn.init.normal_(mlp3.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev3 * S)"""

content = content.replace(
    "            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)\n            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)",
    l3_logic
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_Staggered_L3_Slower_Decay_Stagger")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "from a 3-way split (+0, +6, +12) and added a Layer 3 with even slower decays and +3, +9 stagger.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
