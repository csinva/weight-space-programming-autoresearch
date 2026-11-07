import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# exact ONLY 1.18 = 0.0411
# main ONLY 1.18 = 0.0407
# Both 1.18 = 0.0421
# This proves it's synergistic.
# Let's try Main +1.18, Exact +2.0
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.bias.data[960:988] += 0.82" # 1.18 + 0.82 = 2.0
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Bias_Combo_Exact_2")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Bias Combo patch")
