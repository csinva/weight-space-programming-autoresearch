import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0408.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Add position to exact routing
content = content.replace(
    "        # Write exactly to the last 28 dimensions\n        token_emb[:, 960:988] = token_emb[:, 0:28]",
    "        # Write exactly to the last 28 dimensions\n        token_emb[:, 960:988] = token_emb[:, 0:28]"
)

content = content.replace(
    "            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]",
    "            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]\n            \n        # Add position embeddings to the exact bypass as well (next 2 dimensions)\n        pos_emb[:, 988:990] = pos_emb[:, 28:30]"
)

content = content.replace(
    "model.blocks[0].mlp.fc1.weight.data[:, 960:988] = 0",
    "model.blocks[0].mlp.fc1.weight.data[:, 960:990] = 0"
)
content = content.replace(
    "model.blocks[0].mlp.fc2.weight.data[960:988, :] = 0",
    "model.blocks[0].mlp.fc2.weight.data[960:990, :] = 0"
)
content = content.replace(
    "model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0",
    "model.blocks[1].mlp.fc1.weight.data[:, 960:990] = 0"
)
content = content.replace(
    "model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0",
    "model.blocks[1].mlp.fc2.weight.data[960:990, :] = 0"
)

content = content.replace(
    "Deep_Ensemble_0408_Master",
    "Deep_Ensemble_Exact_Word_Plus_Pos"
)

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact position bypass")
