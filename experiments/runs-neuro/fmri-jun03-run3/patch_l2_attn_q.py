import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# L2 attn Q is currently set to 5.0 (same as L1).
# We tried changing L2 Q to 10.0 and it dropped heavily.
# What about making L2 Q scale softly across networks? Or reducing it?
content = content.replace(
    "l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0",
    "l2_attn.W_q.weight[h_start + 0, d_start + 29] = 3.0"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L2_Q_3_0")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L2 Q 3.0 patch")
