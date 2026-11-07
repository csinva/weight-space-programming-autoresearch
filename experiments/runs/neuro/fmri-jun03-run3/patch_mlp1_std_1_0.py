import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Since tightening MLP1 std dropped the correlation dramatically to 0.0353,
# Let's expand it to 1.0.
content = content.replace(
    "std_dev = 0.5",
    "std_dev = 1.0"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_MLP1_STD_1_0")

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP1 STD 1.0 patch")
