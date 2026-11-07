import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# 1.18 is 0.0421
# 1.15 is 0.0396
# 1.20 is 0.0394
# Let's try 1.17.
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.17"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_LN_Bias_1_17")

with open(filepath, "w") as f:
    f.write(content)
print("Applied LN Bias 1.17 patch")
