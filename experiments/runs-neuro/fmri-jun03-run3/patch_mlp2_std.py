import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We tested MLP1 std and it dropped. Let's test MLP2 std.
# In final_model_0421.py:
# `std_dev2 = 0.01 + float(net) * (3.99 / 14.0)` -> ranges from 0.01 to 4.0
# Let's shift it to range from 0.5 to 5.0.
content = content.replace(
    "std_dev2 = 0.01 + float(net) * (3.99 / 14.0)",
    "std_dev2 = 0.5 + float(net) * (4.5 / 14.0)"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_MLP2_STD_Shift")

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP2 STD Shift patch")
