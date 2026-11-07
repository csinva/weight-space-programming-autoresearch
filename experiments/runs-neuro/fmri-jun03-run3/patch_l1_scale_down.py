import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Orthogonal L1 and L2 dropped performance heavily (to 0.0360).
# This proves that SUPERPOSITION is the key! L1 and L2 being added together in the exact same dimensions is crucial.
# Because L2 overwrites 0-63, and L1 writes to 30-57, they perfectly sum together in the 30-57 block!
# 30-57 is exactly the end of the medium block (22-42) and the long block (43-63)!
# Specifically, L1 (30-57) superimposes on L2's medium block (30-42) and L2's long block (43-57).
# This means the medium and long context embeddings in L2 get a DIRECT INJECTION of sharp local context from L1.

# If this superposition is the magic, how can we tune it?
# We can tune the relative magnitude of the L1 injection!
# Right now, L1 injects into 30-57 with weight `S * 1.0`.
# What if we scale up the L1 injection so the local context is stronger?
# Or scale it down?

# Let's try scaling up the L1 output weight from S * 1.0 to S * 2.0.

replacement = """            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 2.0"""

content = content.replace(
    "            for i in range(28):\n                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0\n                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0",
    replacement
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_L1_Output_Scale_2")
content = content.replace("with L1 decay scale set to 15-80 instead of 10-80.", "with L1 output attention projection scaled by 2.0 to strengthen local superposition.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
