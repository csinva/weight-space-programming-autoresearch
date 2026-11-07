import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0408.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75",
    "l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.76"
)

content = content.replace(
    "Deep_Ensemble_0408_Master",
    "Deep_Ensemble_L1_Scale_176"
)

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Scale 1.76")
