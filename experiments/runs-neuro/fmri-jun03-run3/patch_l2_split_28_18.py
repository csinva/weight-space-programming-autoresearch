import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0407 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0407.py")

with open(filepath, "r") as f:
    content = f.read()

# 18/23/23 dropped heavily (0.0383).
# Let's try 28/18/18 to give maximum space to the local feature.

replacement = """            for i in range(28):
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(18):
                l2_attn.W_v.weight[h_start + 28 + i, d_start_b + i] = 1.0
                l2_attn.W_o.weight[d_start + 28 + i, h_start + 28 + i] = S * 1.0
                
            for i in range(18):
                l2_attn.W_v.weight[h_start + 46 + i, d_start_c + i] = 1.0
                l2_attn.W_o.weight[d_start + 46 + i, h_start + 46 + i] = S * 1.0"""

content = re.sub(
    r'            for i in range\(22\):\n                l2_attn\.W_v\.weight\[h_start \+ i, d_start \+ i\] = 1\.0\n                l2_attn\.W_o\.weight\[d_start \+ i, h_start \+ i\] = S \* 1\.0\n                \n            for i in range\(21\):\n                l2_attn\.W_v\.weight\[h_start \+ 22 \+ i, d_start_b \+ i\] = 1\.0\n                l2_attn\.W_o\.weight\[d_start \+ 22 \+ i, h_start \+ 22 \+ i\] = S \* 1\.0\n                \n            for i in range\(21\):\n                l2_attn\.W_v\.weight\[h_start \+ 43 \+ i, d_start_c \+ i\] = 1\.0\n                l2_attn\.W_o\.weight\[d_start \+ 43 \+ i, h_start \+ 43 \+ i\] = S \* 1\.0',
    replacement,
    content,
    flags=re.DOTALL
)

content = content.replace("Deep_Ensemble_L1_175_L2_Wide", "Deep_Ensemble_L2_Split_28_18_18")
content = content.replace("and L2 bounds widened to 0.01-14.0.", "and L2 dimension split shifted from 22/21/21 to 28/18/18.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
