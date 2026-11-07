import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# 1.7 and 1.8 both got 0.0406!
# This officially breaks the 0.0405 gradient descent trained baseline!!!
# Let's try exactly 1.75 to see if we can squeeze out 0.0407.

replacement = """            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75"""

content = content.replace(
    "            for i in range(28):\n                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0\n                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0",
    replacement
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L1_Output_Scale_1_75")
content = content.replace("with L1 decay scale set to 15-80 instead of 10-80.", "with L1 output attention projection scaled by 1.75.")

with open(filepath, "w") as f:
    f.write(content)

# Save as patch_master_0406.py
os.system("cp runs-neuro/fmri-jun03-run3/interpretable_transformer.py runs-neuro/fmri-jun03-run3/patch_master_0406.py")
print("Updated successfully")
