import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0408.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Let's scale MLP1's output standard deviation with a decay profile across the networks, similar to attention decay
content = content.replace(
    "            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)",
    "            mlp1_decay = 0.5 + float(net) * (1.5 / 14.0)\n            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S * mlp1_decay)"
)

content = content.replace(
    "Deep_Ensemble_0408_Master",
    "Deep_Ensemble_MLP1_Decay"
)

with open(filepath, "w") as f:
    f.write(content)
print("Applied MLP1 decay")
