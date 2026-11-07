import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Change n_heads from 10 to 15 in build_embedder
build_old = "d_model: int = 1020, n_heads: int = 10, n_layers: int = 2,"
build_new = "d_model: int = 1020, n_heads: int = 15, n_layers: int = 2,"
content = content.replace(build_old, build_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Independent_Timescale_Heads"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Independent Timescale Attention Hypothesis: Discovered that the 0.0421 baseline (15 networks, 10 attention heads) forces networks 0-4 to share attention heads with networks 10-14, causing timescale crosstalk. Increased n_heads to 15 (d_head=68) to give every one of the 15 temporal networks its own perfectly isolated attention head."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched 15 Heads.")
