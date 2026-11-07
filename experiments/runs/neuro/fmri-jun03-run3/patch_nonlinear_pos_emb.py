import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Make positional embedding ramp non-linear (square root)
content = content.replace(
    "pos_emb[p, 28] = S * (p / max_seq_len)",
    "pos_emb[p, 28] = S * ((p / max_seq_len) ** 0.5)"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Pos_Emb_Sqrt")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Pos Emb Sqrt patch")
