import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0406 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_l1_scale_1_75.py")

with open(filepath, "r") as f:
    content = f.read()

# We hit 0.0406!
# Now that we've fixed the superposition strength, let's re-test tightening the L2 integration bounds.
# L2 decay is currently 0.05 + float(net) * (11.95 / 14.0) (so 0.05 to 12.0).
# What if we tighten L2 decay to 0.5 to 10.0? So it doesn't do infinitely long decay.
# Let's try 0.5 to 10.0.

replacement = """            # --- LAYER 2: Staggered Integration ---
            l2_decay = 0.5 + float(net) * (9.5 / 14.0)"""

content = content.replace(
    "            # --- LAYER 2: Staggered Integration ---\n            l2_decay = 0.05 + float(net) * (11.95 / 14.0)",
    replacement
)

content = content.replace("Deep_Ensemble_L1_Output_Scale_1_75", "Deep_Ensemble_L1_175_L2_Tight")
content = content.replace("with L1 output attention projection scaled by 1.75.", "with L1 output attention projection scaled by 1.75 and L2 bounds tightened to 0.5-10.0.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
