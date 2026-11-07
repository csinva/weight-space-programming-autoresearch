import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "std_dev2 = 0.01 + float(net) * (3.99 / 14.0)",
    "std_dev2 = 0.05 + float(net) * (7.95 / 14.0)"
)
content = content.replace("Deep_Ensemble_0411_Master", "Deep_Ensemble_MLP2_Scale_Increase")

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP2 scale increase patch")
