import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0408.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Zero out LayerNorm weights for the exact routing dimensions so they literally do not get touched
content = content.replace(
    "            nn.init.ones_(block.ln1.weight)\n            nn.init.zeros_(block.ln1.bias)\n            nn.init.ones_(block.ln2.weight)\n            nn.init.zeros_(block.ln2.bias)",
    "            nn.init.ones_(block.ln1.weight)\n            nn.init.zeros_(block.ln1.bias)\n            nn.init.ones_(block.ln2.weight)\n            nn.init.zeros_(block.ln2.bias)\n            \n            # Don't let LayerNorm touch the exact routing dimensions\n            block.ln1.weight.data[960:988] = 0\n            block.ln2.weight.data[960:988] = 0"
)

content = content.replace(
    "Deep_Ensemble_0408_Master",
    "Deep_Ensemble_Exact_Routing_LN_Zero"
)

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact routing LayerNorm zeroing")
