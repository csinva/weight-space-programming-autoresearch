import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Instead of std=std_dev, lets enforce an exact overlap projection in the MLP
# Let's keep the existing MLP initialization, but add a sparse identity-like pathway
content = content.replace(
    "nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)",
    "nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)\n            for i in range(28):\n                mlp1.fc2.weight[d_start + 30 + i, f_start + i] = S * 0.5"
)
content = content.replace("Deep_Ensemble_0411_Master", "Deep_Ensemble_L1_MLP_Overlap")

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP1 overlap patch")
