import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# I see the MLP class in final_model_0421.py was:
# class MLP(nn.Module):
#    def __init__(self, d_model: int, d_ff: int):
#        super().__init__()
#        self.fc1 = nn.Linear(d_model, d_ff)
#        self.fc2 = nn.Linear(d_ff, d_model)
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        return self.fc2(F.relu(self.fc1(x)))

# Let's replace it safely
mlp_old = """class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))"""

mlp_new = """class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc_gate = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.fc1(x)
        gate = torch.sigmoid(self.fc_gate(x))
        return self.fc2(v * gate)"""

content = content.replace(mlp_old, mlp_new)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Fixed NMDA Gating Patch.")
