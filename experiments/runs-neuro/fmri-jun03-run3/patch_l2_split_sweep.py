import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0407 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0407.py")

with open(filepath, "r") as f:
    content = f.read()

# Current L2 split is 22, 21, 21.
# L1 outputs to 30-57.
# Superposition occurs primarily over the latter half.
# What if we shift the L2 splits to give more or less room to the local features?
# 28, 18, 18
# 18, 23, 23
# Let's try 18, 23, 23.

replacement = """            for i in range(18):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(23):
                l2_attn.W_v.weight[h_start + 18 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 18 + i, h_start + 18 + i] = S * 1.0
                
            for i in range(23):
                l2_attn.W_v.weight[h_start + 41 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 41 + i, h_start + 41 + i] = S * 1.0"""

content = re.sub(
    r'            for i in range\(22\):\n                l2_attn\.W_v\.weight\[h_start \+ i, d_start \+ i\] = 1\.0\n                l2_attn\.W_o\.weight\[d_start \+ i, h_start \+ i\] = S \* 1\.0\n                \n            for i in range\(21\):\n                l2_attn\.W_v\.weight\[h_start \+ 22 \+ i, d_start_b \+ i\] = 1\.0\n                l2_attn\.W_o\.weight\[d_start \+ 22 \+ i, h_start \+ 22 \+ i\] = S \* 1\.0\n                \n            for i in range\(21\):\n                l2_attn\.W_v\.weight\[h_start \+ 43 \+ i, d_start_c \+ i\] = 1\.0\n                l2_attn\.W_o\.weight\[d_start \+ 43 \+ i, h_start \+ 43 \+ i\] = S \* 1\.0',
    replacement,
    content,
    flags=re.DOTALL
)

content = content.replace("Deep_Ensemble_L1_175_L2_Wide", "Deep_Ensemble_L2_Split_18_23_23")
content = content.replace("and L2 bounds widened to 0.01-14.0.", "and L2 dimension split shifted from 22/21/21 to 18/23/23.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
