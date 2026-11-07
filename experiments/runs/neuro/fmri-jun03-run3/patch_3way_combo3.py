import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# Change L1 scale from 10.0-80.0 to 15.0-85.0
content = content.replace("l1_decay = 10.0 + float(net) * (70.0 / 14.0)", "l1_decay = 15.0 + float(net) * (70.0 / 14.0)")

# Change stagger to 0, 7, 14
content = content.replace("net_b = (net + 6) % 15", "net_b = (net + 7) % 15")
content = content.replace("net_c = (net + 12) % 15", "net_c = (net + 14) % 15")

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_Combo3")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with L1 15-85 and 0_7_14 stagger")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
