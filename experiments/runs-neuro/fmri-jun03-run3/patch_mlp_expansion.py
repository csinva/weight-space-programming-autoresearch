import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.blocks[0].mlp.fc2.weight.data *= 1.5\n        model.blocks[1].mlp.fc2.weight.data *= 1.5"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_MLP_FC2_Scale_1_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP fc2 scale 1.5 patch")
