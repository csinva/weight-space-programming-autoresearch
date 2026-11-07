import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Add Astrocyte spatial pooling after MLP
mlp_old = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x"""

mlp_new = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        
        # Astrocyte Glial Network Hypothesis (Spatial Calcium Waves)
        # Astrocytes couple physically adjacent synapses, pooling their activations
        # We model this as a local 1D spatial moving average across the feature dimension (d_ff)
        # kernel size = 5 (coupling 5 neighboring neurons)
        B, T, D_ff = x.shape
        # Reshape for Conv1D: (B*T, 1, D_ff)
        x_flat = x.view(B*T, 1, D_ff)
        # Apply 1D average pooling
        import torch.nn.functional as F
        x_pooled = F.avg_pool1d(x_flat, kernel_size=5, stride=1, padding=2)
        x = x_pooled.view(B, T, D_ff)
        
        x = self.fc2(x)
        return x"""

content = content.replace(mlp_old, mlp_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Astrocyte_Glial_Spatial_Pooling"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Astrocyte Glial Network Hypothesis: Biological computation is not just neuronal; Astrocytes form coupled networks via calcium waves that spatially pool activity across neighboring synapses. Modeled this by inserting a local 1D spatial average-pooling layer (kernel=5) across the hidden neuronal features before projection."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Astrocyte Glial Network.")
