import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0406 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_l1_scale_1_75.py")

with open(filepath, "r") as f:
    content = f.read()

# L2 Tight dropped to 0.0404.
# Let's try L2 Wide: 0.01 to 14.0.

replacement = """            # --- LAYER 2: Staggered Integration ---
            l2_decay = 0.01 + float(net) * (13.99 / 14.0)"""

content = content.replace(
    "            # --- LAYER 2: Staggered Integration ---\n            l2_decay = 0.05 + float(net) * (11.95 / 14.0)",
    replacement
)

content = content.replace("Deep_Ensemble_L1_Output_Scale_1_75", "Deep_Ensemble_L1_175_L2_Wide")
content = content.replace("with L1 output attention projection scaled by 1.75.", "with L1 output attention projection scaled by 1.75 and L2 bounds widened to 0.01-14.0.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
