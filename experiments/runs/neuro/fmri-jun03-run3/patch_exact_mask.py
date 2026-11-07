import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Since we established 0.0421 is the hard peak, I want to understand what the exact tokens are actually doing.
# If we replace the exact tokens with Gaussian noise at the output, what happens?
content = content.replace(
    "return x",
    "noise = torch.randn_like(x[:, :, 960:988])\n        x[:, :, 960:988] = noise\n        return x"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Exact_Noise")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Exact Noise patch")
