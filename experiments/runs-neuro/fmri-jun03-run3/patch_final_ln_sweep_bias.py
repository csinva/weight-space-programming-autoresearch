import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.25"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Final_LN_Bias_1_25")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Final LN bias 1.25 patch")
