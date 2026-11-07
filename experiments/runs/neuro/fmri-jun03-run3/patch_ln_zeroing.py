import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We tried scaling exact tokens and it dropped.
# Let's try zeroing out the exact tokens completely in the final output to see if they are doing anything at all.
# If it drops, they are useful.
content = content.replace(
    "x = self.final_ln(x)",
    "x = self.final_ln(x)\n        x[:, :, 960:988] = 0.0"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Zero_Exact_Tokens")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Zero Exact Tokens patch")
