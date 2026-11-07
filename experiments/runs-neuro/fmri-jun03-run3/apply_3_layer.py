filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

# Change n_layers default
content = content.replace("n_layers: int = 2", "n_layers: int = 3")

# Zero out MLP for layer 3
content = content.replace(
    "model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0",
    "model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0\n        model.blocks[2].mlp.fc1.weight.data[:, 960:988] = 0\n        model.blocks[2].mlp.fc2.weight.data[960:988, :] = 0"
)

# Update name and description
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_3Layer_Depth")
content = content.replace(
    "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.",
    "Adds a 3rd Transformer Layer to the optimal 0.0421 baseline. Layer 3 applies an extremely slow, deeply staggered integration (decays 0.001-2.0) of Layer 2's phrase-level features to build sentence-level macro-structures."
)

with open(filepath, "w") as f:
    f.write(content)
