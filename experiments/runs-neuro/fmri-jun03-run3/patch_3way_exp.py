import argparse
import sys
import os
import re
import math

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# First, restore to the Masterpiece (3Way 0_6_12 with 0.0398)
os.system("python3 runs-neuro/fmri-jun03-run3/patch_ultra_tune_final.py")

with open(filepath, "r") as f:
    content = f.read()

# Make sure math is imported
if "import math" not in content:
    content = "import math\n" + content

# Replace the L2 decay with exponential 0.01 to 15.0
l2_exp_code = """
            # L2 exponential decay
            l2_log_start = math.log(0.01)
            l2_log_end = math.log(15.0)
            log_val2 = l2_log_start + (float(net) / 14.0) * (l2_log_end - l2_log_start)
            l2_decay = math.exp(log_val2)
"""
content = re.sub(
    r'            l2_decay = 0\.05 \+ float\(net\) \* \(11\.95 \/ 14\.0\)',
    l2_exp_code,
    content,
    flags=re.DOTALL
)

# Update name and description
content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way", "Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_ExpL2")
content = content.replace("from a 2-way split (+0, +7) to a 3-way split (+0, +6, +12)", "from a 3-way split (+0, +6, +12) with purely exponential L2 decay from 0.01 to 15.0")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
