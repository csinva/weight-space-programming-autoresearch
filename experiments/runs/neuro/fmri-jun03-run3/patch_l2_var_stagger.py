import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# In 0.0399, L2 variance is: std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
# But it applies this uniformly to all 64 dims of the net's MLP.
# This means the local feature (0-21), medium feature (22-42), and long feature (43-63) all get the SAME variance!
# What if we explicitly scale them so the longest context has the highest variance?
# Wait, we tried 1.1/1.2 multipliers before and got 0.0399.
# What if we scale the medium and long features by a function of the NET?

# Or what if we change the MLP completely? What if we don't have standard normal init, but Kaiming or Xavier?
# The zero-shot interpretable transformer uses nn.init.normal_ with specific std_dev.
# What if we just increase the std_dev2 range? It is 0.01 to 4.0.
# Let's try 0.01 to 8.0! Kicking up the variance for the longest timescales.

replacement = """            std_dev2 = 0.01 + float(net) * (7.99 / 14.0)"""

content = content.replace(
    "            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)",
    replacement
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L2_Var_8_0")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "from a 3-way split (+0, +6, +12) with L2 MLP variance increased to 8.0.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
