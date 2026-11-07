import os
import sys

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

# Change n_layers from 2 to 1
content = content.replace("n_layers: int = 2", "n_layers: int = 1")
content = content.replace("Deep_Ensemble_0421_Master", "Single_Layer_Simplicity")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "Removing Layer 2 to test if a single layer can maintain the 0.0421 performance, purely for interpretability and simplicity.")

with open(filepath, "w") as f:
    f.write(content)
