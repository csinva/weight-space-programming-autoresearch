import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Add a slight variance scale to L1 output projection
content = content.replace(
    "                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75",
    "                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * (1.75 + float(net) * (0.25 / 14.0))"
)

content = content.replace(
    "Deep_Ensemble_0411_Master",
    "Deep_Ensemble_L1_Var_Inj"
)

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Var Inj")
