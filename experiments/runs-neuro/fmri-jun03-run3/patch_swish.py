import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Replace ReLU with Swish/SiLU in the MLP block
content = content.replace(
    "return self.fc2(F.relu(self.fc1(x)))",
    "return self.fc2(F.silu(self.fc1(x)))"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Swish")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Swish patch")
