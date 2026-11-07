filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "l2_attn = model.blocks[1].attn" in line:
        skip = True
    if skip and "# Zero out MLP and Attention for dimensions 960-988" in line:
        skip = False
        
    if not skip:
        if "model.blocks[1].mlp" not in line:
            new_lines.append(line)

with open(filepath, "w") as f:
    f.writelines(new_lines)
