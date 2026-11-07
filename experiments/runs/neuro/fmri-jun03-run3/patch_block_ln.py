import os
import sys

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
bias_val = float(sys.argv[1])

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

old_block_ln = """        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)"""

new_block_ln = f"""        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            block.ln1.bias.data += {bias_val}
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
            block.ln2.bias.data += {bias_val}"""

if old_block_ln not in content:
    print("Error: Could not find old block ln logic to replace.")
    exit(1)

content = content.replace(old_block_ln, new_block_ln)
content = content.replace("Deep_Ensemble_0421_Master", f"Deep_Ensemble_Block_LN_{bias_val}")

with open(filepath, "w") as f:
    f.write(content)
print(f"Applied block LN bias patch {bias_val}")
