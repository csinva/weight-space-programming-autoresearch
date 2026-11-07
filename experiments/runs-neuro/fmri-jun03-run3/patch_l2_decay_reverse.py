import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Current:
#             # --- LAYER 2: Staggered Integration ---
#             l2_decay = 0.05 + float(net) * (11.95 / 14.0)

# Replace with reversed:
replacement = """            # --- LAYER 2: Staggered Integration ---
            l2_decay = 12.0 - float(net) * (11.95 / 14.0)"""

content = content.replace(
    "            # --- LAYER 2: Staggered Integration ---\n            l2_decay = 0.05 + float(net) * (11.95 / 14.0)",
    replacement
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L2_Decay_Reversed")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "from a 3-way split (+0, +6, +12) with L2 decay bounds reversed (12.0 down to 0.05).")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
