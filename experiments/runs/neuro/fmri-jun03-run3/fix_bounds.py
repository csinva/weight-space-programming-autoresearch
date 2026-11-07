filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

content = content.replace("d_model: int = 1020", "d_model: int = 1024")
content = content.replace("n_heads: int = 10", "n_heads: int = 16")

with open(filepath, "w") as f:
    f.write(content)
