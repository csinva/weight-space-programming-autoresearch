import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

replacement = """
    def forward(self, idx):
        x = self.word_emb(idx)
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        x = x + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_ln(x)
        # Manually subtract the 1.18 bias from the exact token dimensions so they stay raw
        x[:, :, 960:988] -= 1.18
        return x
"""

content = re.sub(r'    def forward\(self, idx\):\n        x = self.word_emb\(idx\)\n        pos = torch.arange\(0, x.size\(1\), dtype=torch.long, device=x.device\)\n        x = x \+ self.pos_emb\(pos\)\n        \n        for block in self.blocks:\n            x = block\(x\)\n            \n        x = self.final_ln\(x\)\n        return x', replacement, content)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Exact_Tokens_No_Bias")

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact tokens no bias patch")
