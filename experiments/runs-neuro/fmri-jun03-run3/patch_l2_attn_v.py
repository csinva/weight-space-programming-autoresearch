import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We played with bias/scales. Let's see if we can improve L2 V matrix.
# V matrix values are 1.0. Let's try 1.5.
# Wait, let's scale the output projection `W_o` values globally for L2!
content = content.replace(
    "model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0",
    "model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0\n        model.blocks[1].attn.W_o.weight.data *= 1.2"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L2_W_o_Scale_1_2")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L2 W_o Scale 1.2 patch")
