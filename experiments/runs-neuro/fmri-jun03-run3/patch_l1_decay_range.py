import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# L1 decay currently scales from 15 to 80 (`15.0 + float(net) * (65.0 / 14.0)`)
# Let's try scaling from 20 to 120 (sharper locals)
content = content.replace(
    "l1_decay = 15.0 + float(net) * (65.0 / 14.0)",
    "l1_decay = 20.0 + float(net) * (100.0 / 14.0)"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L1_Decay_Sharp")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Decay Sharp patch")
