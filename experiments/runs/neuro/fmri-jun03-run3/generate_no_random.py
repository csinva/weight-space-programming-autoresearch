import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

# Replace the random normal initialization with deterministic orthogonal mixing
pattern = r"std_dev = 0\.5.*?std_dev \* S\)"
replacement = """
            std_dev = 0.5
            torch.manual_seed(42 + net)
            nn.init.orthogonal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], gain=std_dev)
            nn.init.orthogonal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], gain=std_dev * S)
"""
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

pattern2 = r"std_dev2 = 0\.01 \+ float\(net\) \* \(3\.99 / 14\.0\).*?std_dev2 \* S\)"
replacement2 = """
            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            torch.manual_seed(100 + net)
            nn.init.orthogonal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], gain=std_dev2)
            nn.init.orthogonal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], gain=std_dev2 * S)
"""
content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_Orthogonal_Subspace"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "Replaced random normal MLP with strict orthogonal matrices.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)
