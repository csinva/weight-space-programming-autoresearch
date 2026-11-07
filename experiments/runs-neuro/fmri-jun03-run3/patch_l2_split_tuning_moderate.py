import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0409.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.25\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0\n                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0\n                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 0.75",
    "                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.15\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0\n                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0\n                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 0.85"
)

content = content.replace(
    "Deep_Ensemble_0409_Master",
    "Deep_Ensemble_L2_Split_Tuning_Moderate"
)

with open(filepath, "w") as f:
    f.write(content)
print("Applied L2 Split Tuning Moderate")
