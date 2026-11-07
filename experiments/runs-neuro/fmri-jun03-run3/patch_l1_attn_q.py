import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# L1 Q is currently 5.0. 
# We had 0.0421.
# What if we scale it gently? e.g. 6.0 or 4.0? Let's try 6.0
content = content.replace(
    "l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0",
    "l1_attn.W_q.weight[h_start + 0, d_start + 29] = 6.0"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L1_Q_6_0")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Q 6.0 patch")
