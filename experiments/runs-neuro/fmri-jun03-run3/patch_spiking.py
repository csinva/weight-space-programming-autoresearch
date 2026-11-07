import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify MLP activation
mlp_old = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x"""

mlp_new = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        # Spiking Action Potential: Binary thresholding instead of continuous rate-coding (ReLU)
        # Neurons fire (1) if voltage > 0, else stay quiet (0).
        x = (x > 0.0).float() 
        x = self.fc2(x)
        return x"""

content = content.replace(mlp_old, mlp_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Binary_Spiking_Action_Potentials"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Spiking Neural Network Hypothesis: Replaces the continuous rate-coding activation (ReLU) with binary action potentials (Hard Step Function). Tests if the brain\'s representation is better modeled by discrete all-or-nothing spikes rather than continuous firing rates."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Spiking Action Potentials.")
