import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Exact noise: 0.0358 (from 0.0421).
# What if we replace main features (0-960) with noise?
content = content.replace(
    "return x",
    "noise = torch.randn_like(x[:, :, :960])\n        x[:, :, :960] = noise\n        return x"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Main_Noise")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Main Noise patch")
