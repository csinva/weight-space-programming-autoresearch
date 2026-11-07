import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# All recent variations failed to break 0.0399.
# Exact word routing hit 0.0397.
# We know 0.0399 is extremely robust.
# Let's try combining the two best things:
# 1. 15-80 L1 bound
# 2. Widening the L2 bounds significantly!
# We previously tried widening L2 bounds to 0.01 to 15.0 and it hit 0.0396.
# What about a smaller widening: 0.01 to 12.0 is current (0.05 + 11.95).
# Let's try 0.05 to 14.0.

replacement = """            # --- LAYER 2: Staggered Integration ---
            l2_decay = 0.05 + float(net) * (13.95 / 14.0)"""

content = content.replace(
    "            # --- LAYER 2: Staggered Integration ---\n            l2_decay = 0.05 + float(net) * (11.95 / 14.0)",
    replacement
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L2_Scale_14_0")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "from a 3-way split (+0, +6, +12) with L2 decay scale pushed from 12.0 max to 14.0 max.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
