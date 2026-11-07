import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398/0.0399)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Change dim_per_net to 68
content = content.replace("dim_per_net = 64", "dim_per_net = 68")

# Adjust the splits in L2 from 22/21/21 to 23/23/22 = 68
split_code = """            for i in range(23):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(23):
                l2_attn.W_v.weight[h_start + 23 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 23 + i, h_start + 23 + i] = S * 1.0
                
            for i in range(22):
                l2_attn.W_v.weight[h_start + 46 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 46 + i, h_start + 46 + i] = S * 1.0"""

content = re.sub(
    r'            for i in range\(22\):.*?l2_attn\.W_o\.weight\[d_start \+ 43 \+ i, h_start \+ 43 \+ i\] = S \* 1\.0',
    split_code,
    content,
    flags=re.DOTALL
)

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_Dim68")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80", "from a 3-way split (+0, +6, +12) but utilizing all 1020 dimensions (dim_per_net=68)")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
