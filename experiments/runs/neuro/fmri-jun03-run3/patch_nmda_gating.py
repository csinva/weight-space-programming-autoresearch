import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify MLP to include NMDA Multiplicative Gating (GLU)
mlp_old = """class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x"""

mlp_new = """class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc_gate = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Compartment Dendritic Gating (NMDA receptors)
        # Replaces simple ReLU with a multiplicative gate (Gated Linear Unit)
        # Activation requires both presynaptic input (fc1) AND dendritic depolarization (fc_gate)
        v = self.fc1(x)
        gate = torch.sigmoid(self.fc_gate(x))
        x = v * gate
        x = self.fc2(x)
        return x"""

content = content.replace(mlp_old, mlp_new)


# Update write_weights to initialize fc_gate
# We'll just clone the fc1 initialization block for fc_gate
ww_old1 = """            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)"""

ww_new1 = """            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc_gate.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)"""

content = content.replace(ww_old1, ww_new1)

ww_old2 = """            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)"""

ww_new2 = """            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc_gate.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)"""

content = content.replace(ww_old2, ww_new2)

ww_old3 = """        # Zero out MLP and Attention for dimensions 960-988 to keep exact tokens pure
        model.blocks[0].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[0].mlp.fc2.weight.data[960:988, :] = 0
        model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0"""

ww_new3 = """        # Zero out MLP and Attention for dimensions 960-988 to keep exact tokens pure
        model.blocks[0].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[0].mlp.fc_gate.weight.data[:, 960:988] = 0
        model.blocks[0].mlp.fc2.weight.data[960:988, :] = 0
        model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[1].mlp.fc_gate.weight.data[:, 960:988] = 0
        model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0"""

content = content.replace(ww_old3, ww_new3)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Multi_Compartment_NMDA_Gating"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Multi-Compartment Dendritic Gating (NMDA Receptor) Hypothesis: Biological neurons do not just sum inputs linearly and apply a threshold (ReLU). Dendrites have complex multi-compartment interactions, specifically NMDA receptors which act as coincidence detectors (multiplicative gating). Replaced ReLU with a Gated Linear Unit (v * sigmoid(gate)) to test if non-linear multiplicative interaction is required to map linguistic semantics to BOLD signals."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched NMDA Gating.")
