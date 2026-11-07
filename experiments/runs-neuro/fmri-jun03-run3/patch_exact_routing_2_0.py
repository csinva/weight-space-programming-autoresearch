import os
import sys

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0408.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Scale the routing to 2.0 to see if it helps
content = content.replace(
    "token_emb[:, 960:988] = token_emb[:, 0:28]",
    "token_emb[:, 960:988] = token_emb[:, 0:28] * 2.0"
)

content = content.replace(
    "Deep_Ensemble_0408_Master",
    "Deep_Ensemble_Exact_Routing_2_0"
)

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact routing scale 2.0")
