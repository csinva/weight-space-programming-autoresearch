import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We scaled W_o for L2 to 1.2 (0.0394) and 0.8 (0.0396).
# Let's try scaling W_o for L1 down slightly.
content = content.replace(
    "model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0",
    "model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0\n        model.blocks[0].attn.W_o.weight.data *= 0.8"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L1_W_o_0_8")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 W_o 0.8 patch")
