import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Since we had success shifting biases later, what if we shift word embeddings?
# Wait, we tried shifting word embeddings bias and it broke things.
# What if we scale word embeddings BEFORE adding pos emb?
content = content.replace(
    "x = self.word_emb(idx)",
    "x = self.word_emb(idx)\n        x = x * 1.5"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Word_Emb_Scale_1_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Word Emb Scale 1.5 patch")
