import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "nn.init.zeros_(block.ln1.bias)",
    "nn.init.zeros_(block.ln1.bias)\n            block.ln1.bias.data += 1.0"
)
content = content.replace(
    "nn.init.zeros_(block.ln2.bias)",
    "nn.init.zeros_(block.ln2.bias)\n            block.ln2.bias.data += 1.0"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Block_LN_Shift_Pos_1_0")

with open(filepath, "w") as f:
    f.write(content)
print("Applied block ln positive shift patch")
