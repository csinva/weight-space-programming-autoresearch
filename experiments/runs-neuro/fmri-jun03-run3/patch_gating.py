import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Add a gating mechanism to the MLP
new_mlp_forward = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GLU-style gating mechanism without adding new weights
        # We split the hidden dimension in half
        h = self.fc1(x)
        d_ff_half = h.shape[-1] // 2
        gate = torch.sigmoid(h[..., :d_ff_half])
        val = F.relu(h[..., d_ff_half:])
        gated_h = gate * val
        # To match expected shapes, we pad with zeros or just double
        gated_h = torch.cat([gated_h, gated_h], dim=-1)
        return self.fc2(gated_h)"""

content = re.sub(
    r"    def forward\(self, x: torch\.Tensor\) -> torch\.Tensor:\n        return self\.fc2\(F\.relu\(self\.fc1\(x\)\)\)",
    new_mlp_forward,
    content
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_MLP_Gating")

with open(filepath, "w") as f:
    f.write(content)
print("Applied Gating patch")
