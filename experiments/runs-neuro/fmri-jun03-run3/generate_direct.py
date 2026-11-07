import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

# We will remove layer 2 and just have 1 layer that mixes delays
# To test if hierarchical structure matters at all
content = content.replace("l2_attn = model.blocks[1].attn", "l2_attn = model.blocks[0].attn")
content = content.replace("mlp2 = model.blocks[1].mlp", "mlp2 = model.blocks[0].mlp")
content = content.replace("n_layers: int = 2", "n_layers: int = 1")

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_1_Layer_Direct"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "Ablated the model down to a single layer to test if hierarchical integration actually adds value.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)
