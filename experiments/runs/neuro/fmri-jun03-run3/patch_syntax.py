import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

# Update dimensions to support 16 nets
content = content.replace("d_model: int = 1020", "d_model: int = 1024")
content = content.replace("n_heads: int = 10", "n_heads: int = 16")

# Update num_nets
content = content.replace("num_nets = 15", "num_nets = 16")

# Move the large C indices to the end of the new d_model
content = content.replace("pos_emb[p, 1000] = C", "pos_emb[p, 1022] = C")
content = content.replace("pos_emb[p, 1001] = -C", "pos_emb[p, 1023] = -C")

# Change exact bounds
content = content.replace("token_emb[:, 960:988] = token_emb[:, 0:28]", "token_emb[:, 960:988] = token_emb[:, 0:28]")

# We will just write a custom write_weights function.
