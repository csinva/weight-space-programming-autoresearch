import os
import sys

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
scale = sys.argv[1]

os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

old_forward = """    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return x"""

new_forward = f"""    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.final_ln(x) + x * {scale}"""

content = content.replace(old_forward, new_forward)
content = content.replace("Deep_Ensemble_0421_Master", f"Deep_Ensemble_Final_Skip_{scale}")

with open(filepath, "w") as f:
    f.write(content)
print(f"Applied final LN skip patch {scale}")
