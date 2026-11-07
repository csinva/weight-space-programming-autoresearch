import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# Change L2 decay from 0.05 to 12.0 -> 0.01 to 15.0
content = content.replace("l2_decay = 0.05 + float(net) * (11.95 / 14.0)", "l2_decay = 0.01 + float(net) * (14.99 / 14.0)")

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L2_Wider")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with L2 decay widened from 0.05-12.0 to 0.01-15.0 to capture extremely long horizons")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
