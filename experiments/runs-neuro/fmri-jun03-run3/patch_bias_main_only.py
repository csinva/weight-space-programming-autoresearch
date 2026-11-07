import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Only on exact tokens got 0.0411.
# What about ONLY on main features (0-960)?
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data[:960] += 1.18"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Bias_Main_Only")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Bias Main Only patch")
