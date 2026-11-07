import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# It's `token_emb` which is set via `model.token_emb.weight.data.copy_(token_emb)`
content = content.replace(
    "model.token_emb.weight.data.copy_(token_emb)",
    "token_emb[:, 960:988] *= 1.5\n        model.token_emb.weight.data.copy_(token_emb)"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Exact_Word_Emb_1_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact token emb patch")
