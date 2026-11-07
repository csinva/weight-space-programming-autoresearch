import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Let's try std 0.4
content = content.replace(
    "std_dev = 0.5",
    "std_dev = 0.4"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_MLP1_STD_0_4")

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP1 STD 0.4 patch")
