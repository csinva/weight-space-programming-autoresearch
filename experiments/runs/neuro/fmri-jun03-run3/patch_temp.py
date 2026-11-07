import os
import sys

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
temp_scale = sys.argv[1]

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Replace scores scale
content = content.replace(
    "scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)",
    f"scores = (q @ k.transpose(-2, -1)) / (math.sqrt(dh) * {temp_scale})"
)
content = content.replace("Deep_Ensemble_0421_Master", f"Deep_Ensemble_Temp_{temp_scale}")

with open(filepath, "w") as f:
    f.write(content)
print(f"Applied Temp patch {temp_scale}")
