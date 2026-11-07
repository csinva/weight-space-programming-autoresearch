import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# 15 to 80 (baseline): 0.0421
# 10 to 60: 0.0405
# 5 to 40: 0.0407
# Let's try 12 to 70.
content = content.replace(
    "l1_decay = 15.0 + float(net) * (65.0 / 14.0)",
    "l1_decay = 12.0 + float(net) * (58.0 / 14.0)"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L1_Decay_12_70")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Decay 12 to 70 patch")
