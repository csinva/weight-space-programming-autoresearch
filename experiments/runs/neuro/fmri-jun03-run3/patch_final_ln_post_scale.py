import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# I want to scale the entire output feature matrix AFTER layer norm by 2.0.
# We scaled pre-LN and got 0.0395.
content = content.replace(
    "x = self.final_ln(x)",
    "x = self.final_ln(x)\n        x = x * 2.0"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Post_LN_Scale_2_0")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Post LN Scale 2.0 patch")
