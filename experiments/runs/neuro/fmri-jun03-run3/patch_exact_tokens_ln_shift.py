import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.bias.data[960:988] -= 0.5"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Exact_Tokens_LN_Shift")

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact tokens ln shift patch")
