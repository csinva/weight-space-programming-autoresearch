import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# OH MY GOD. 0.0404!!!
# We are 0.0001 away from the gradient descent baseline (0.0405)!
# The superposition strength was the missing variable! L1 injecting its sharp local context into L2's medium/long context was being suppressed.
# Let's try pushing it to 3.0!

replacement = """            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 3.0"""

content = content.replace(
    "            for i in range(28):\n                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0\n                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0",
    replacement
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L1_Output_Scale_3")
content = content.replace("with L1 decay scale set to 15-80 instead of 10-80.", "with L1 output attention projection scaled by 3.0 to heavily strengthen local superposition.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
