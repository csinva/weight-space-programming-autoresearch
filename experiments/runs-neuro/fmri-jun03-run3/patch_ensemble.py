import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

with open(filepath, "r") as f:
    content = f.read()

content = content.replace("for p in models[0].parameters():", "for p in model.parameters():")

with open(filepath, "w") as f:
    f.write(content)
