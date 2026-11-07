import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Let's give the exact tokens a dedicated positional embedding scale.
content = content.replace(
    "model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0",
    "model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0\n        # Add a tiny bit of pos emb to exact tokens\n        for p in range(max_seq_len):\n            model.pos_emb.weight.data[p, 960:988] = S * (p / max_seq_len) * 0.1"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Exact_Pos_Emb")

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact pos emb patch")
