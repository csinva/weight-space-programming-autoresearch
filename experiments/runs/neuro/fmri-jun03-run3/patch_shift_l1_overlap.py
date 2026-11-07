import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0408 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0408.py")

with open(filepath, "r") as f:
    content = f.read()

# Right now L1 outputs to 30-57 (28 dims).
# L2 splits are 0-21 (local), 22-42 (medium), 43-63 (long).
# This means L1 overlaps with 30-42 (medium) and 43-57 (long).
# What if we shift L1 to start at 22 so it perfectly overlaps the ENTIRE medium context and the beginning of the long context?
# 22 + 28 = 50. So it would overlap 22-42 (medium) and 43-50 (long).

replacement = """            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 22 + i, h_start + i] = S * 1.75"""

content = content.replace(
    "            for i in range(28):\n                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0\n                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75",
    replacement
)

content = content.replace("Deep_Ensemble_0408_Master", "Deep_Ensemble_L1_Overlap_Shift_22")
content = content.replace("with L1 output attention projection scaled by 1.75, L2 bounds widened to 0.01-14.0, and pure exact token routing in 960-988.", "with L1 output shifted to dims 22-49 to perfectly overlap the medium context.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
