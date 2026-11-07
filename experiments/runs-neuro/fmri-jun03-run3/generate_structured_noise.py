import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

# Replace the random normal initialization with structured hash functions

pattern = r"std_dev = 0\.5.*?std_dev \* S\)"

replacement = """
            # Replace random noise with structured polynomial projections
            for out_f in range(ff_per_net):
                for in_d in range(dim_per_net):
                    # Deterministic hash function to simulate random projection but structured
                    val1 = math.sin((out_f * 31.0 + in_d * 17.0 + net * 43.0) * 0.1) * 0.5
                    val2 = math.cos((out_f * 29.0 + in_d * 23.0 + net * 41.0) * 0.1) * 0.5 * S
                    mlp1.fc1.weight[f_start + out_f, d_start + in_d] = val1
                    mlp1.fc2.weight[d_start + in_d, f_start + out_f] = val2
"""

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

pattern2 = r"std_dev2 = 0\.01 \+ float\(net\) \* \(3\.99 / 14\.0\).*?std_dev2 \* S\)"

replacement2 = """
            std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            for out_f in range(ff_per_net):
                for in_d in range(dim_per_net):
                    val1 = math.sin((out_f * 37.0 + in_d * 19.0 + net * 47.0) * 0.1) * std_dev2
                    val2 = math.cos((out_f * 43.0 + in_d * 13.0 + net * 53.0) * 0.1) * std_dev2 * S
                    mlp2.fc1.weight[f_start + out_f, d_start + in_d] = val1
                    mlp2.fc2.weight[d_start + in_d, f_start + out_f] = val2
"""

content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_Deterministic_Subspace"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "Replaced random normal MLP subspaces with deterministic mathematical projections.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)
