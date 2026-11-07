import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0408.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# We discovered L1 overlaps tightly with L2. Let's see what happens if we shift L1 to map to dims 31-58 (just a small +1 shift).
content = content.replace(
    "                l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75",
    "                l1_attn.W_o.weight[d_start + 31 + i, h_start + i] = S * 1.75"
)

content = content.replace(
    "Deep_Ensemble_0408_Master",
    "Deep_Ensemble_L1_Overlap_Shift_31"
)

with open(filepath, "w") as f:
    f.write(content)
print("Applied L1 Overlap Shift 31")
