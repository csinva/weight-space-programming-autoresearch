import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We tried pos emb sqrt -> 0.0400
# Can we combine pos emb sqrt with LN Bias 1.18? We just did that.
# What about a smaller LN Bias 1.18 with a specific shift on exact tokens? No, that also failed.

# Let's verify our baseline again to make sure 0.0421 is still the absolute peak, but let's test L2 `W_o` scale downwards.
content = content.replace(
    "model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0",
    "model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0\n        model.blocks[1].attn.W_o.weight.data *= 0.8"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L2_W_o_0_8")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L2 W_o 0.8 patch")
