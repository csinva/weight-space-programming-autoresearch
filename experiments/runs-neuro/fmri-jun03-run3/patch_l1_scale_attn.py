import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Scale the final attention representation coming out of L1.
content = content.replace(
    "x = x + self.attn(self.ln1(x))",
    "x = x + self.attn(self.ln1(x)) * 1.5"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L1_Attn_Scale_1_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Attn Scale 1.5 patch")
