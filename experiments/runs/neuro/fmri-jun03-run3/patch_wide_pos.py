import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0399 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_3way_l1_scale_15_80.py")

with open(filepath, "r") as f:
    content = f.read()

# Position embeddings are used to drive the attention decay mechanism.
# Positional differences drive the (q @ k) dot product.
# We are currently doing:
#             pos_emb[p, 28] = S * (p / max_seq_len)
#             pos_emb[p, 29] = S * 1.0

# What if we scale the positional differences up or down?
# If we scale p / max_seq_len by 5.0, then the positional distances will be larger, which will make the attention drop off faster.
# But we ALREADY tune the decay rates (L1 is 15-80, L2 is 0.05-12.0) to counteract whatever positional scale we have.
# But a larger positional scale increases the resolution / dynamic range relative to any noise added by the model.

# Let's try changing the positional scalar.

replacement = """        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            # Increase dynamic range of position encoding
            pos_emb[p, 28] = S * (p / max_seq_len) * 2.0
            pos_emb[p, 29] = S * 1.0"""

content = content.replace(
    "        for p in range(max_seq_len):\n            pos_emb[p, 1000] = C\n            pos_emb[p, 1001] = -C\n            \n            pos_emb[p, 28] = S * (p / max_seq_len)\n            pos_emb[p, 29] = S * 1.0",
    replacement
)

content = content.replace("Deep_Ensemble_Staggered_Asymmetric_UltraTune_3Way_L1_Scale_15_80", "Deep_Ensemble_Pos_Scale_2")
content = content.replace("with L1 decay scale set to 15-80 instead of 10-80.", "with position encoding dynamic range doubled.")

with open(filepath, "w") as f:
    f.write(content)
print("Updated successfully")
