import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "token_emb[:, 960:988] = token_emb[:, 0:28]",
    "token_emb[:, 960:988] = token_emb[:, 0:28] * 1.5"
)

content = content.replace(
    "Deep_Ensemble_0411_Master",
    "Deep_Ensemble_L2_Split_Tuning_Exact_1_5"
)

with open(filepath, "w") as f:
    f.write(content)
print("Applied Exact Routing 1.5")
