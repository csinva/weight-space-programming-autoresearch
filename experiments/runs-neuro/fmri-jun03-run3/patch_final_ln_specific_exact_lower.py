import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We tried removing the +1.18 on exact tokens completely before (it dropped to 0.0401)
# What if we increase the shift on exact tokens to +1.5, while keeping L1/L2 at 1.18?
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.bias.data[960:988] += 0.32" # 1.5 total
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Final_LN_Bias_Exact_1_50")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Final LN bias exact patch 1.50")
