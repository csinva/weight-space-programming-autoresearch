import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Add a pre-LN scale to the exact tokens
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
        
        # Give exact tokens a multiplier before normalization so they take up more "volume"
        # of the normalized space. Exact tokens are in dims 960-988.
        x_scaled = x.clone()
        x_scaled[:, :, 960:988] = x_scaled[:, :, 960:988] * 2.0
        
        x = self.final_ln(x_scaled)
        return x"""

if old_forward not in content:
    print("Error: Could not find old forward logic to replace.")
    exit(1)

content = content.replace(old_forward, new_forward)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Exact_PreScale_2.0")

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact prescale patch")
