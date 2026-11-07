import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0408 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0408.py")

with open(filepath, "r") as f:
    content = f.read()

# Shifting to 22-49 completely broke the model (0.0350).
# The overlap with the long-context block (43-63) is apparently CRITICAL.
# 30-57 overlaps with 13 dims of medium and 15 dims of long.
# What if we shift L1 completely into the long-context block?
# Long context is 43-63 (21 dims). L1 needs 28 dims.
# So we can't fit L1 entirely in the long context.
# But we can shift it to 36-63 (28 dims).
# This perfectly overlaps the second half of medium (36-42) and all of long (43-63).

replacement = """            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 36 + i, h_start + i] = S * 1.75"""

content = content.replace(
    "            for i in range(28):\n                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0\n                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75",
    replacement
)

content = content.replace("Deep_Ensemble_0408_Master", "Deep_Ensemble_L1_Overlap_Shift_36")
content = content.replace("with L1 output attention projection scaled by 1.75, L2 bounds widened to 0.01-14.0, and pure exact token routing in 960-988.", "with L1 output shifted to dims 36-63 to perfectly overlap the long context.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
