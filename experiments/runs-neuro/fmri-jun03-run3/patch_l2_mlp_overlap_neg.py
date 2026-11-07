import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)",
    "nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)\n            for i in range(22):\n                mlp2.fc2.weight[d_start + i, f_start + i] = S * -0.2"
)
content = content.replace("Deep_Ensemble_0411_Master", "Deep_Ensemble_L2_MLP_Overlap_Neg")

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP2 neg overlap patch")
