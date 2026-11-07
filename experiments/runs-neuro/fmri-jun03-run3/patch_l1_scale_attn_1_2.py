import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# 1.5 dropped to 0.0399. Let's try 1.2.
content = content.replace(
    "x = x + self.attn(self.ln1(x))",
    "x = x + self.attn(self.ln1(x)) * 1.2"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L1_Attn_Scale_1_2")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Attn Scale 1.2 patch")
