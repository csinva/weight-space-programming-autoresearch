import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "x = x + self.pos_emb(pos)",
    "x = x + self.pos_emb(pos) * 1.5"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Pos_Emb_Scale_1_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Pos Emb Scale 1.5 patch")
