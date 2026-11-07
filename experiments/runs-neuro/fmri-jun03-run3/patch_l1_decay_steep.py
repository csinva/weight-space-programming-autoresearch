import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Make the local extraction even sharper
content = content.replace(
    "l1_decay = 15.0 + float(net) * (65.0 / 14.0)",
    "l1_decay = 25.0 + float(net) * (75.0 / 14.0)"
)
content = content.replace("Deep_Ensemble_0411_Master", "Deep_Ensemble_L1_Decay_25_100")

with open(filepath, "w") as f:
    f.write(content)
print("Applied steep L1 decay")
