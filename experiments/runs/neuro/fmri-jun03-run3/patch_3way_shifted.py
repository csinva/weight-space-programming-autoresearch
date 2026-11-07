import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# Change stagger to +4, +8
content = content.replace("net_b = (net + 6) % 15", "net_b = (net + 4) % 15")
content = content.replace("net_c = (net + 12) % 15", "net_c = (net + 8) % 15")

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_0_4_8")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) to (+0, +4, +8)")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
