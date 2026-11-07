import torch
import math
import torch.nn as nn
import os
import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Instead of 'pass', use 'pass' properly
content = re.sub(r"nn\.init\.normal_\(mlp1\.fc1\.weight.*?\)", "pass", content)
content = re.sub(r"nn\.init\.normal_\(mlp1\.fc2\.weight.*?\)", "pass", content)
content = re.sub(r"nn\.init\.normal_\(mlp2\.fc1\.weight.*?\)", "pass", content)
content = re.sub(r"nn\.init\.normal_\(mlp2\.fc2\.weight.*?\)", "pass", content)

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_No_Noise_Ablation"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "Ablation test: removing all random normal noise from the MLP blocks.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer_temp.py", "w") as f:
    f.write(content)
