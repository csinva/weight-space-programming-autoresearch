import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Since LN bias 1.18 is extremely specific, let's see if injecting a tiny bias into word embeddings helps?
# We did pos emb scale, word emb scale.
content = content.replace(
    "x = self.word_emb(idx)",
    "x = self.word_emb(idx) + 0.1"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Word_Emb_Bias_0_1")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Word Emb Bias 0.1 patch")
