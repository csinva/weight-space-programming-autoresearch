import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

# We need to test if the random noise in MLP needs MORE dimensionality.
# The SOTA uses ff_per_net = 256. 15 * 256 = 3840. We have d_ff=4000 available.
# But what if we expanded d_ff to 16000?

content = content.replace("ff_per_net = 256", "ff_per_net = 1000")
content = content.replace("f_start = 15 * 256", "f_start = 15 * 1000")
content = content.replace("d_ff: int = 4000", "d_ff: int = 16000")

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_Wide_Random_Subspace"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "Expanded the random normal MLP subspace projection from d_ff=4000 to d_ff=16000.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)
