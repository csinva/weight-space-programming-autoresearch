import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Instead of LayerNorm post-shift, what if we scale the inputs BEFORE LayerNorm?
# We can do this by scaling all outputs from MLP2, or scaling in the forward pass.
content = content.replace(
    "x = self.final_ln(x)",
    "x = x * 2.0\n        x = self.final_ln(x)"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Pre_LN_Scale_2_0")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Pre LN Scale 2.0 patch")
