import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# 0.68 on L1/L2 main features got 0.0418! That's very close to 0.0421.
# What if we go even lower on the main features, keeping the 1.18 on the exact tokens?
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.bias.data[:960] -= 0.8" # 1.18 - 0.8 = 0.38
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Final_LN_Bias_Long_0_38")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Final LN bias specific patch lower more")
