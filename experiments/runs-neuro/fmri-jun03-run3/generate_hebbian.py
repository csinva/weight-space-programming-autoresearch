import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

# Replace the random normal initialization with Hebbian-like associative memory
# Instead of purely random, we make the projection weights heavily block-diagonal, grouping letters

pattern = r"std_dev = 0\.5.*?std_dev \* S\)"
replacement = """
            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            
            # Make fc2 symmetric to fc1 to act as an auto-encoder memory!
            mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net] = mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net].T * S
"""
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

pattern2 = r"std_dev2 = 0\.01 \+ float\(net\) \* \(3\.99 / 14\.0\).*?std_dev2 \* S\)"
replacement2 = """
            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net] = mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net].T * S
"""
content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_AutoEncoder_Subspace"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "Tied MLP fc1 and fc2 weights symmetrically to act as associative auto-encoders.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)
