import os
import sys

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Change d_model to 2040, n_heads to 20
old_builder = """def build_embedder(device: str = 'cuda',
                   d_model: int = 1020, n_heads: int = 10, n_layers: int = 2,
                   d_ff: int = 4000, max_seq_len: int = 64) -> InterpretableEmbedder:"""

new_builder = """def build_embedder(device: str = 'cuda',
                   d_model: int = 2040, n_heads: int = 20, n_layers: int = 2,
                   d_ff: int = 8000, max_seq_len: int = 64) -> InterpretableEmbedder:"""

# Change num_nets to 30
old_params = """        num_nets = 15
        dim_per_net = 64
        ff_per_net = 256"""

new_params = """        num_nets = 30
        dim_per_net = 64
        ff_per_net = 256"""

# Change staggered logic for 30 nets
old_staggered = """            # Staggered logic: 3-way split
            net_b = (net + 6) % 15
            net_c = (net + 12) % 15"""

new_staggered = """            # Staggered logic: 3-way split
            net_b = (net + 10) % 30
            net_c = (net + 20) % 30"""

# Change decay scaling factors for 30 nets
content = content.replace("float(net) * (65.0 / 14.0)", "float(net) * (65.0 / 29.0)")
content = content.replace("float(net) * (13.99 / 14.0)", "float(net) * (13.99 / 29.0)")
content = content.replace("float(net) * (3.99 / 14.0)", "float(net) * (3.99 / 29.0)")

# Fix exact tokens dimensions (now they should be in the remaining dims)
# 30 * 64 = 1920. Remaining dims = 2040 - 1920 = 120.
# Let's map exact tokens to 1920:1948
old_exact1 = """        # Write exactly to the last 28 dimensions
        token_emb[:, 960:988] = token_emb[:, 0:28]"""
new_exact1 = """        # Write exactly to the last 28 dimensions
        token_emb[:, 1920:1948] = token_emb[:, 0:28]"""

old_exact2 = """        # Zero out MLP and Attention for dimensions 960-988 to keep exact tokens pure
        model.blocks[0].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[0].mlp.fc2.weight.data[960:988, :] = 0
        model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0"""
new_exact2 = """        # Zero out MLP and Attention for dimensions to keep exact tokens pure
        model.blocks[0].mlp.fc1.weight.data[:, 1920:1948] = 0
        model.blocks[0].mlp.fc2.weight.data[1920:1948, :] = 0
        model.blocks[1].mlp.fc1.weight.data[:, 1920:1948] = 0
        model.blocks[1].mlp.fc2.weight.data[1920:1948, :] = 0"""

content = content.replace(old_builder, new_builder)
content = content.replace(old_params, new_params)
content = content.replace(old_staggered, new_staggered)
content = content.replace(old_exact1, new_exact1)
content = content.replace(old_exact2, new_exact2)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Double_Dim_2040")

with open(filepath, "w") as f:
    f.write(content)
print("Applied double dim patch")
