import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Change n_layers from 2 to 3
content = content.replace("n_layers: int = 2", "n_layers: int = 3")

l3_defs = """        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        l3_attn = model.blocks[2].attn
        mlp3 = model.blocks[2].mlp
        
        for net in range(num_nets):"""

content = content.replace(
    "        l2_attn = model.blocks[1].attn\n        mlp2 = model.blocks[1].mlp\n        \n        for net in range(num_nets):",
    l3_defs
)

# This time, we don't mix dimensions in L3. We just let L3 *further* decay the L2 representations (which are ALREADY mixed!)
# This is a true cascade. L1 extracts exactly -> L2 mixes horizontally across timescales -> L3 vertically smooths the mixtures.

l3_logic = """            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)
            
            # --- LAYER 3: Pure Cascade Smoothing ---
            # Don't stagger again. Just take the already-staggered representations from L2 and smooth them more.
            # Use L2-like decay rates (they are very slow: 0.05 to 12.0)
            l3_decay = 0.05 + float(net) * (11.95 / 14.0)
            
            l3_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l3_attn.W_k.weight[h_start + 0, d_start + 28] = l3_decay
            
            for i in range(dim_per_net - 30):
                l3_attn.W_v.weight[h_start + i, d_start + 30 + i] = 1.0
                l3_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
                
            std_dev3 = 0.01 + float(net) * (3.99 / 14.0)
            nn.init.normal_(mlp3.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev3)
            nn.init.normal_(mlp3.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev3 * S)"""

content = content.replace(
    "            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)\n            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)",
    l3_logic
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L3_Pure_Cascade")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "from a 3-way split (+0, +6, +12) but adding a Layer 3 pure cascade smoothing step without re-staggering.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
