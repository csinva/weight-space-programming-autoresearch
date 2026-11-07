import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Change n_heads to 5 in the builder
content = content.replace("n_heads: int = 10", "n_heads: int = 5")
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_5_Heads")

with open(filepath, "w") as f:
    f.write(content)
print("Applied 5 heads patch")
