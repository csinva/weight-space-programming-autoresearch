import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system(f"cp runs-neuro/fmri-jun03-run3/final_model_0421.py {filepath}")

with open(filepath, "r") as f:
    content = f.read()

content = content.replace("d_model: int = 1020", "d_model: int = 2040")
content = content.replace("n_heads: int = 10", "n_heads: int = 20")
content = content.replace("d_ff: int = 4000", "d_ff: int = 8000")
content = content.replace("Deep_Ensemble_0421_Master", "Massive_Scale_2040_8000")

with open(filepath, "w") as f:
    f.write(content)
print("Applied massive scale patch")
