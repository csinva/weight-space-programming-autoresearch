import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# We need to insert a bottleneck between Layer 1 and Layer 2.
# We can do this in the `forward` pass of the main model or Block 0.
# Since the model uses a standard `Block`, let's just modify the `SimpleTransformer.forward`.

model_old = """        for block in self.blocks:
            x = block(x)"""

model_new = """        # Global Workspace Theory (GWT) Bottleneck between Layer 1 and Layer 2
        # Layer 1
        x = self.blocks[0](x)
        
        # The Global Workspace Bottleneck (Compression to 16 dimensions and back)
        # We simulate this without trainable parameters by just using a random projection matrix
        # Or, we can just do an SVD-like hard truncation if we had weights, but we can do a fixed random projection.
        B, T, D = x.shape
        # Squeeze to 16 dims (the Workspace)
        workspace = x @ self.W_compress.to(x.device) # (B, T, 16)
        # Broadcast back to D dims
        x = workspace @ self.W_broadcast.to(x.device) # (B, T, D)
        
        # Layer 2
        x = self.blocks[1](x)"""

content = content.replace(model_old, model_new)

# Now we need to add W_compress and W_broadcast to SimpleTransformer
init_old = """        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)"""

init_new = """        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        
        # Global Workspace matrices
        workspace_dim = 16
        # Orthogonal random projections
        import torch.nn.init as init
        self.W_compress = nn.Parameter(torch.empty(d_model, workspace_dim), requires_grad=False)
        self.W_broadcast = nn.Parameter(torch.empty(workspace_dim, d_model), requires_grad=False)
        init.orthogonal_(self.W_compress)
        # Broadcast is the transpose to perfectly reconstruct what's possible
        self.W_broadcast.data = self.W_compress.data.t()"""

content = content.replace(init_old, init_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Global_Workspace_Theory"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Global Workspace Theory (GWT) Hypothesis: Tests if the 15 parallel temporal networks operate completely independently, or if they must broadcast their state through a central, low-dimensional bottleneck (the Global Workspace). Added a harsh 16-dimensional bottleneck between Layer 1 and Layer 2."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Global Workspace Theory.")
