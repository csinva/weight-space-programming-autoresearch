import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# We want to add the explicit, un-integrated word vectors into the L2 output so the linear ridge regression gets direct access to the raw word tokens as well as the integrated context.
# Right now, L2 receives the L1 output and mixes it. 
# The initial token embeddings are at dims 0-27. We can map them straight through.
# In L1: map 0-27 straight through.
# In L2: map 0-27 straight through.
# Wait, L1 and L2 already map 0-27 straight through!
# L1: l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0 (for i in 0 to 27)
# L2: wait, L2 maps 0-21, 22-42, 43-63. It completely mangles the first 28 dims (0-27).
# Let's see L2 logic.

# In L1:
#             for i in range(28):
#                 l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
#                 l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
# Wait, in L1, the output is placed at d_start + 30.
# The original token embeddings were at d_start 0-27.
# And because of the residual connection `x = x + self.attn(self.ln1(x))`, the original token embeddings at 0-27 are still there!
# Then in L2, l2_attn maps from input dims 0-21 (which are the original tokens), and places them at output 0-21.
# And from d_start_b + 0-20 to output 22-42.
# So L2 IS acting on the original tokens (0-21), not the L1 outputs (30-57)!
# WOW.

# If L2 is acting on 0-21, it is ignoring the L1 output (30-57) entirely in the attention projection!
# Wait. Is L2 attention completely ignoring L1's attention output??
# Let's check l2_attn.W_v.weight
