import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We applied L1 exponential decay.
# Let's see if we can apply an exponential decay to L2 attention scores.
# In final_model_0421.py, we have `attn2 = apply_decay_l2(attn2)`. Do we?
# Wait, let's look at what we actually have in final_model_0421.py for L2.
