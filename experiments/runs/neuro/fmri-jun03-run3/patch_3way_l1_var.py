import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# Change L1 std_dev from constant 0.5 to graded 0.1 to 1.0
content = content.replace("std_dev = 0.5", "std_dev = 0.1 + float(net) * (0.9 / 14.0)")

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Graded")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with graded L1 variance (0.1 to 1.0) rather than fixed (0.5)")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
