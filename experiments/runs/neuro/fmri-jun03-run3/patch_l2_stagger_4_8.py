import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "net_b = (net + 6) % 15\n            net_c = (net + 12) % 15",
    "net_b = (net + 4) % 15\n            net_c = (net + 8) % 15"
)
content = content.replace("Deep_Ensemble_0411_Master", "Deep_Ensemble_L2_Stagger_4_8")

with open(filepath, "w") as f:
    f.write(content)
print("Applied stagger 4/8 patch")
