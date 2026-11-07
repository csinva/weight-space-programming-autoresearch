import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify MLP to include stochastic synaptic failure (Dropout forced on during eval)
mlp_old = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x"""

mlp_new = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        
        # Stochastic Synaptic Failure Hypothesis (Vesicle Release Probability)
        # Biological synapses are highly unreliable; an action potential only triggers
        # neurotransmitter vesicle release ~50% of the time.
        # We simulate this by forcing a 50% dropout mask on the activations,
        # actively dropping signals randomly even during inference/evaluation.
        import torch.nn.functional as F
        x = F.dropout(x, p=0.5, training=True) 
        
        x = self.fc2(x)
        return x"""

content = content.replace(mlp_old, mlp_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Stochastic_Synaptic_Failure"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Stochastic Synaptic Failure Hypothesis: Biological synapses are notoriously unreliable, with vesicle release probabilities often around 50%. Tested if this noise acts as a robust regularizer (like biological dropout) for the semantic representations by forcing a 50% random signal dropout during the forward pass of evaluation."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Synaptic Failure.")
