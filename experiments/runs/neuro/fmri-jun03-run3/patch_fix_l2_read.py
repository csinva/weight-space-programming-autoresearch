import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Currently in L2:
#             for i in range(22):
#                 l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
#                 l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
#                 
#             for i in range(21):
#                 l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
#                 l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
#                 
#             for i in range(21):
#                 l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
#                 l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0

# This means L2 is reading the raw tokens (0-21) and NOT the L1 output (30-57).
# L1 output is at dims 30-57 (which has 28 dimensions).
# If we want L2 to integrate the L1 features, it should read from 30-51, 30-50, 30-50 for example.
# Actually, the fact that L2 was reading the raw tokens directly means L1 and L2 were operating independently!
# L1 was doing fast decay, L2 was doing slow decay. They were parallel networks! 
# The MLP in L2 was receiving BOTH the L1 outputs (in 30-57 via residual) AND the L2 outputs (in 0-63).
# Wait, L2 overwrites 0-63 via residual stream! So L2 output adds to the residual stream.
# Residual at L2 output:
# dims 0-21: raw tokens + L2 slow integration from net
# dims 22-42: raw tokens + L2 slow integration from net_b
# dims 43-63: raw tokens (or zero) + L2 slow integration from net_c
# dims 30-57: L1 fast integration + (whatever was just added).

# If L1 and L2 are actually parallel features, making them sequential might change everything.
# Let's try making L2 ACTUALLY sequential, by having it read the L1 outputs instead of raw tokens!

replacement = """            for i in range(22):
                # Read from L1 output (starts at 30)
                l2_attn.W_v.weight[h_start + i, d_start + 30 + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + 30 + i] = 1.0
                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + 30 + i] = 1.0
                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0"""

content = content.replace(
    "            for i in range(22):\n                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0\n                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0\n                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0\n                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0",
    replacement
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L2_Reads_L1")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "from a 3-way split (+0, +6, +12). Modifies L2 attention to read from L1 outputs (dims 30+) rather than raw tokens (dims 0+).")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
