import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "model.blocks[0].mlp.fc1.weight.data[:, 960:988] = 0",
    "model.blocks[0].mlp.fc1.weight.data[:, 900:988] = 0"
)
content = content.replace(
    "model.blocks[0].mlp.fc2.weight.data[960:988, :] = 0",
    "model.blocks[0].mlp.fc2.weight.data[900:988, :] = 0"
)
content = content.replace(
    "model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0",
    "model.blocks[1].mlp.fc1.weight.data[:, 900:988] = 0"
)
content = content.replace(
    "model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0",
    "model.blocks[1].mlp.fc2.weight.data[900:988, :] = 0"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Exact_Tokens_900")

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact tokens 900 patch")
