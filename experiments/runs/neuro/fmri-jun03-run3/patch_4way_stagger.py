import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0408 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0408.py")

with open(filepath, "r") as f:
    content = f.read()

# We tried 4-way earlier and it failed. BUT that was before we fixed the L1 overlap superposition and exact word routing!
# Now that superposition is tuned, what if a 4-way split works better?
# 4-way split of 64 dimensions: 16, 16, 16, 16.
# 15 nets. Step = 15/4 ~ 4. Let's use 0, 4, 8, 12.

replacement_1 = """            # Staggered logic: 4-way split
            net_b = (net + 4) % 15
            net_c = (net + 8) % 15
            net_d = (net + 12) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            d_start_d = net_d * dim_per_net
            
            for i in range(16):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(16):
                l2_attn.W_v.weight[h_start + 16 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 16 + i, h_start + 16 + i] = S * 1.0
                
            for i in range(16):
                l2_attn.W_v.weight[h_start + 32 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 32 + i, h_start + 32 + i] = S * 1.0
                
            for i in range(16):
                l2_attn.W_v.weight[h_start + 48 + i, d_start_d + i] = 1.0
                l2_attn.W_o.weight[d_start + 48 + i, h_start + 48 + i] = S * 1.0"""

content = re.sub(
    r'            # Staggered logic: 3-way split.*?l2_attn\.W_o\.weight\[d_start \+ 43 \+ i, h_start \+ 43 \+ i\] = S \* 1\.0',
    replacement_1,
    content,
    flags=re.DOTALL
)

content = content.replace("Deep_Ensemble_0408_Master", "Deep_Ensemble_4Way_Stagger")
content = content.replace("pure exact token routing in 960-988.", "pure exact token routing in 960-988, and a 4-way stagger (16/16/16/16).")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
