import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Instead of bias, let's play with weight. Bias shift 1.18 worked wonders.
# Let's scale the weight too. Default is 1.0. Let's try 1.5, 0.5.
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.weight.data *= 0.5"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_LN_Scale_0_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied LN Scale 0.5 patch")
