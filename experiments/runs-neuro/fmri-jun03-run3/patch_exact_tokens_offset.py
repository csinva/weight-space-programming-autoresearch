import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Currently dims 960-988 are the pure word tokens. Let's see if adding standard deviation or altering bounds matters.
# Let's change the exact tokens window to 950-988, increasing the window size
content = content.replace(
    "model.blocks[0].mlp.fc1.weight.data[:, 960:988] = 0",
    "model.blocks[0].mlp.fc1.weight.data[:, 950:988] = 0"
)
content = content.replace(
    "model.blocks[0].mlp.fc2.weight.data[960:988, :] = 0",
    "model.blocks[0].mlp.fc2.weight.data[950:988, :] = 0"
)
content = content.replace(
    "model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0",
    "model.blocks[1].mlp.fc1.weight.data[:, 950:988] = 0"
)
content = content.replace(
    "model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0",
    "model.blocks[1].mlp.fc2.weight.data[950:988, :] = 0"
)

content = content.replace("Deep_Ensemble_0411_Master", "Deep_Ensemble_Exact_Tokens_950")

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact tokens patch")
