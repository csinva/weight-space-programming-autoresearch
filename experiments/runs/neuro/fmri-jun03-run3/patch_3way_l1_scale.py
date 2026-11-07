import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# Change L1 scale from 10.0-80.0 to 15.0-90.0
content = content.replace("l1_decay = 10.0 + float(net) * (70.0 / 14.0)", "l1_decay = 15.0 + float(net) * (75.0 / 14.0)")

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with L1 decay scale tightened to 15-90 instead of 10-80")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
