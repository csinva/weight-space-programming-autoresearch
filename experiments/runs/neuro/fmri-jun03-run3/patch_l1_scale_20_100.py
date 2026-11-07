import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

replacement = """            # --- LAYER 1: Extremely sharp local extraction ---
            l1_decay = 20.0 + float(net) * (80.0 / 14.0)"""

content = content.replace(
    "            # --- LAYER 1: Extremely sharp local extraction ---\n            l1_decay = 15.0 + float(net) * (65.0 / 14.0)",
    replacement
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L1_Scale_20_100")
content = content.replace("with L1 decay scale set to 15-80 instead of 10-80.", "with L1 decay scale set to 20-100.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
