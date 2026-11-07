import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.blocks[1].mlp.fc2.bias.data += 0.5"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_MLP2_FC2_Bias_0_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP2 FC2 bias 0.5 patch")
