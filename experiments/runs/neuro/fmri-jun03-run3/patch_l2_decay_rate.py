import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# L2 decay rate is set to `l2_decay = 0.01 + float(net) * (13.99 / 14.0)` which spans 0.01 to 14.0.
# Let's try narrowing it or shifting it. Say, 0.01 to 5.0.
content = content.replace(
    "l2_decay = 0.01 + float(net) * (13.99 / 14.0)",
    "l2_decay = 0.01 + float(net) * (4.99 / 14.0)"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L2_Decay_5_0")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L2 Decay 5.0 patch")
