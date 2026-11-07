import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0408 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0408.py")

with open(filepath, "r") as f:
    content = f.read()

# Fix the math import issue. math is already imported at the top of the file!
# The problem is that importing math inside write_weights shadows the module-level math!

replacement = """            std_dev2 = 0.01 * math.exp(float(net) * (math.log(4.0 / 0.01) / 14.0))"""

content = content.replace(
    "            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)",
    replacement
)

content = content.replace("Deep_Ensemble_0408_Master", "Deep_Ensemble_L2_Var_Exp")
content = content.replace("with L1 output attention projection scaled by 1.75, L2 bounds widened to 0.01-14.0, and pure exact token routing in 960-988.", "with L2 MLP std scaled exponentially from 0.01 to 4.0.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
