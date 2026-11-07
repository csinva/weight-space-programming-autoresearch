import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# Combine:
# 1. 26/19/19 feature split (gave 0.0394 by itself, worth trying with other combos) - wait, let's keep 22/21/21 for now since 26/19/19 dropped it slightly from 0.0398
# Let's combine:
# 1. L1 scale 15-90 (gave 0.0398, tied for record)
# 2. L2 scale 0.01-15.0 (gave 0.0396 by itself, strong)
# 3. L1 variance graded 0.1-1.0 (gave 0.0369, NEVER MIND, do not use)
# 4. Standard 0, 6, 12 stagger

# Change L1 scale from 10.0-80.0 to 15.0-90.0
content = content.replace("l1_decay = 10.0 + float(net) * (70.0 / 14.0)", "l1_decay = 15.0 + float(net) * (75.0 / 14.0)")

# Change L2 decay from 0.05-12.0 to 0.01-15.0
content = content.replace("l2_decay = 0.05 + float(net) * (11.95 / 14.0)", "l2_decay = 0.01 + float(net) * (14.99 / 14.0)")

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_Combo")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with combined optimal L1 scale (15-90) and widened L2 decay (0.01-15.0)")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
