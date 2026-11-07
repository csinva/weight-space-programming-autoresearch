import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "model.pos_emb.weight.data.copy_(pos_emb)",
    "model.pos_emb.weight.data.copy_(pos_emb)\n        model.pos_emb.weight.data += 0.5" 
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Pos_Emb_Shift_0_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied pos emb shift patch")
