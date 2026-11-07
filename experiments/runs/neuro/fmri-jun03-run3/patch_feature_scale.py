import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Try scaling exact tokens significantly after layernorm (or just giving them an explicit bias after)
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
        x = self.final_ln(x)
        
        # After LayerNorm, dynamically boost the Exact Tokens
        # Because ridge regression is very sensitive to feature magnitude,
        # bumping these specific dimensions might give the regression a hint
        # exact tokens are in 960:988
        
        # We need to clone to avoid in-place modification errors
        out = x.clone()
        out[:, :, 960:988] = out[:, :, 960:988] * 2.0
        return out"""

content = content.replace(old_forward, new_forward)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Post_Scale_Exact_2")

with open(filepath, "w") as f:
    f.write(content)
print("Applied post-scale exact tokens patch")
