import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Add a skip connection bypassing the final layernorm entirely
old_forward = """    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return x"""

new_forward = """    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        # Skip connection: return the normalized output PLUS the raw unnormalized output
        return self.final_ln(x) + x * 0.1"""

if old_forward not in content:
    print("Error: Could not find old forward logic to replace.")
    exit(1)

content = content.replace(old_forward, new_forward)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Final_Skip_0.1")

with open(filepath, "w") as f:
    f.write(content)
print("Applied final LN skip patch")
