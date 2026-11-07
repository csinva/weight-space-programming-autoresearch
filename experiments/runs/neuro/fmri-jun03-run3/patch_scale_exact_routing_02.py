import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0408 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0408.py")

with open(filepath, "r") as f:
    content = f.read()

# Scale 5.0 dropped to 0.0406.
# Maybe we should scale it down? Let's try 0.2.

replacement = """        # Write exactly to the last 28 dimensions
        token_emb[:, 960:988] = token_emb[:, 0:28] * 0.2"""

content = content.replace(
    "        # Write exactly to the last 28 dimensions\n        token_emb[:, 960:988] = token_emb[:, 0:28]",
    replacement
)

content = content.replace("Deep_Ensemble_0408_Master", "Deep_Ensemble_Exact_Routing_Scale_0_2")
content = content.replace("pure exact token routing in 960-988.", "pure exact token routing in 960-988 scaled by 0.2.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
