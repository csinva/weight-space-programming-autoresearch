import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Baseline 15 to 80 is 0.0421. We tested ranges lower.
# Let's test a range slightly higher: 18 to 90.
content = content.replace(
    "l1_decay = 15.0 + float(net) * (65.0 / 14.0)",
    "l1_decay = 18.0 + float(net) * (72.0 / 14.0)"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L1_Decay_18_90")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Decay 18 to 90 patch")
