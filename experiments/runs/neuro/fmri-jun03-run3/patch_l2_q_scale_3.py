import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0",
    "l2_attn.W_q.weight[h_start + 0, d_start + 29] = 3.0"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_L2_Q_Scale_3")

with open(filepath, "w") as f:
    f.write(content)
print("Applied L2 Q scale 3 patch")
