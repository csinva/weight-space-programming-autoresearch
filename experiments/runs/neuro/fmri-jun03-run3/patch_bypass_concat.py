import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Instead of relying on regex replacements that might hit multiple `forward` functions (e.g. MLP, Attention),
# let's only replace inside SimpleTransformer.
content = re.sub(
    r'(def forward\(self, idx: torch\.Tensor\) -> torch\.Tensor:.*?x = self\.word_emb\(idx\).*?)return x',
    r'\1x_out = self.final_ln(x)\n        return x_out + self.word_emb(idx)',
    content,
    flags=re.DOTALL
)

content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Add_Raw_Emb")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Add Raw Emb fix 4 patch")
