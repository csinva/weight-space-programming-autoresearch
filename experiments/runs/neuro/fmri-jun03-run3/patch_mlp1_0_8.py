import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We tested MLP1 std 0.2 (0.0353) and 1.0 (0.0362). Baseline is 0.5 (0.0421).
# What about std 0.8?
content = content.replace(
    "std_dev = 0.5",
    "std_dev = 0.8"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_MLP1_STD_0_8")

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP1 STD 0.8 patch")
