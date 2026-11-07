import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# We discovered that L1 writes to 30-57, and L2 reads 0-21.
# So L1 and L2 are actually operating completely in parallel.
# L2 is getting the raw words (0-21) and creating a slow-decay context embedding in 0-63.
# L1 is getting the raw words (0-27) and creating a fast-decay local word embedding in 30-57.
# Then they are both just concatenated together in the residual stream!
# Because L2 overwrites 0-63, wait...
# L2 W_v mapped to 0-63.
# BUT DOES L2 OVERWRITE L1?
# Let's see: L1 wrote to 30-57.
# L2 writes to 0-21, 22-42, 43-63.
# If L2 writes to 30-57, it ADDS to it!
# Because of residual connection! x = x + attn(x).
# So in dims 30-42, we have: L1 fast decay + L2 slow decay from net B.
# In dims 43-57, we have: L1 fast decay + L2 slow decay from net C.
# So L1 and L2 are literally just being linearly added together in the residual stream!
# What if we explicitly stop L2 from overlapping with L1?

# If L2 writes to 60-120? We don't have enough dimensions (only 64 per net).
# But wait, L1 writes to 30-57 (28 dims).
# L2 writes to 0-63 (64 dims).
# So L1 and L2 are summed in 30-57.
# What if we just reduce L2 to write to 0-29?
# No, we need 3-way staggering.
# What if we expand dim_per_net to 100?

replacement = """        num_nets = 10  # Reduced to fit 100 dims per net in 1020 total dims
        dim_per_net = 100
        ff_per_net = 400"""

content = re.sub(
    r'        num_nets = 15\n        dim_per_net = 64\n        ff_per_net = 256',
    replacement,
    content,
    flags=re.DOTALL
)

# And now L1 writes to 70-97
l1_replace = """            for i in range(28):
                l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l1_attn.W_o.weight[d_start + 70 + i, h_start + i] = S * 1.0"""

content = re.sub(
    r'            for i in range\(28\):\n                l1_attn\.W_v\.weight\[h_start \+ i, d_start \+ i\] = 1\.0\n                l1_attn\.W_o\.weight\[d_start \+ 30 \+ i, h_start \+ i\] = S \* 1\.0',
    l1_replace,
    content,
    flags=re.DOTALL
)

# And L2 writes to 0-63
l2_replace = """            # Staggered logic: 3-way split
            net_b = (net + 4) % 10
            net_c = (net + 8) % 10"""

content = re.sub(
    r'            # Staggered logic: 3-way split\n            net_b = \(net \+ 6\) % 15\n            net_c = \(net \+ 12\) % 15',
    l2_replace,
    content,
    flags=re.DOTALL
)

# Wait! The 3-way split in L2 relies on ranges: 0-21, 22-42, 43-63.
# Which totals 64 dims. So it fits cleanly in 0-63, completely avoiding L1's 70-97!
# We also have to fix l1_decay and l2_decay because there are only 10 nets now.

decay_replace_1 = """            # --- LAYER 1: Extremely sharp local extraction ---
            l1_decay = 15.0 + float(net) * (65.0 / 9.0) """
content = content.replace("            # --- LAYER 1: Extremely sharp local extraction ---\n            l1_decay = 15.0 + float(net) * (65.0 / 14.0) ", decay_replace_1)

decay_replace_2 = """            # --- LAYER 2: Staggered Integration ---
            l2_decay = 0.05 + float(net) * (11.95 / 9.0)"""
content = content.replace("            # --- LAYER 2: Staggered Integration ---\n            l2_decay = 0.05 + float(net) * (11.95 / 14.0)", decay_replace_2)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_Orthogonal_L1_L2")
content = content.replace("from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80.", "by expanding to 100 dims per net (10 nets) to make L1 output (70-97) orthogonal to L2 output (0-63).")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
