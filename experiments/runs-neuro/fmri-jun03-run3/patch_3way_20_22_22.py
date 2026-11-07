import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# Change feature splitting from 22, 21, 21 to 20, 22, 22
content = content.replace("for i in range(22):", "for i in range(20):")
content = content.replace("for i in range(21):\n                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0\n                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0", "for i in range(22):\n                l2_attn.W_v.weight[h_start + 20 + i, d_start_b + i] = 1.0\n                l2_attn.W_o.weight[d_start + 20 + i, h_start + 20 + i] = S * 1.0")
content = content.replace("for i in range(21):\n                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0\n                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0", "for i in range(22):\n                l2_attn.W_v.weight[h_start + 42 + i, d_start_c + i] = 1.0\n                l2_attn.W_o.weight[d_start + 42 + i, h_start + 42 + i] = S * 1.0")

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_20_22_22")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split with 22/21/21 features to 20/22/22 features (giving more weight to the distant timescales)")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
