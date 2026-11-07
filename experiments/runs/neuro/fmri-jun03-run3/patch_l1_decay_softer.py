import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Soft range 5 to 40 got 0.0407! Which is very good (close to 0.0421).
# Let's try 10 to 60.
content = content.replace(
    "l1_decay = 15.0 + float(net) * (65.0 / 14.0)",
    "l1_decay = 10.0 + float(net) * (50.0 / 14.0)"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L1_Decay_10_60")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Decay 10 to 60 patch")
