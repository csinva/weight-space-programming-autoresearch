import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# The error happened because W_compress and W_broadcast were likely added in the wrong init or were overwritten.
# Let's check where they were added.
# Ah, I replaced the text `self.blocks = ...` but maybe it was inside `__init__` or somewhere else?
with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Ah! My previous replace failed because the text `self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])`
# was actually spread across multiple lines in the original file:
# self.blocks = nn.ModuleList([
#    Block(d_model, n_heads, d_ff) for _ in range(n_layers)
# ])

init_old = """        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)"""

init_new = """        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        
        workspace_dim = 16
        import torch.nn.init as init
        import torch
        self.W_compress = nn.Parameter(torch.empty(d_model, workspace_dim), requires_grad=False)
        self.W_broadcast = nn.Parameter(torch.empty(workspace_dim, d_model), requires_grad=False)
        init.orthogonal_(self.W_compress)
        self.W_broadcast.data = self.W_compress.data.t()"""

content = content.replace(init_old, init_new)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Fixed GWT Patch.")
