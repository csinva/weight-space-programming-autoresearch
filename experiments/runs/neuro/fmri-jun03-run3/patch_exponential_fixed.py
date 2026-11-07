import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# Replace both L1 and L2 with correctly implemented exponential spacing
# L1 from 10 to 80 exponentially
# L2 from 0.05 to 12.0 exponentially
exp_code = """
            l1_log_start = math.log(10.0)
            l1_log_end = math.log(80.0)
            log_val = l1_log_start + (float(net) / 14.0) * (l1_log_end - l1_log_start)
            l1_decay = math.exp(log_val)
"""

l2_exp_code = """
            l2_log_start = math.log(0.05)
            l2_log_end = math.log(12.0)
            log_val2 = l2_log_start + (float(net) / 14.0) * (l2_log_end - l2_log_start)
            l2_decay = math.exp(log_val2)
"""

content = re.sub(
    r'            l1_decay = 10\.0 \+ float\(net\) \* \(70\.0 \/ 14\.0\)',
    exp_code,
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'            l2_decay = 0\.05 \+ float\(net\) \* \(11\.95 \/ 14\.0\)',
    l2_exp_code,
    content,
    flags=re.DOTALL
)

# Make sure math is imported correctly
if "import math" not in content:
    content = "import math\n" + content

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_FullyExp")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with correctly implemented full exponential decay spacing for both L1 and L2")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
