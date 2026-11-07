import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We tried removing LN bias on exact tokens and it dropped.
# We tried scaling exact tokens and it dropped.
# Let's bypass the LayerNorm entirely for exact tokens.
content = content.replace(
    "x = self.final_ln(x)",
    "x_norm = self.final_ln(x)\n        x_norm[:, :, 960:988] = x[:, :, 960:988]\n        x = x_norm"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_LN_Bypass")

with open(filepath, "w") as f:
    f.write(content)
print("Applied LN Bypass patch")
