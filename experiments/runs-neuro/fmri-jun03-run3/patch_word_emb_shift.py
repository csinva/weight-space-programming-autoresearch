import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "model.word_emb.weight.data.copy_(word_embeddings_tensor)",
    "model.word_emb.weight.data.copy_(word_embeddings_tensor)\n        model.word_emb.weight.data += 0.5" 
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Word_Emb_Shift_0_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied word emb shift patch")
