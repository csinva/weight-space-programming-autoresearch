import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

pattern = r"        # LINGUISTIC DETECTORS \(Head 9\).*?h_start \+ i\] = S \* 1\.0"
content = re.sub(pattern, "        pass", content, flags=re.DOTALL)

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_Pure_Staggered"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "Ablated the 11 morphological trackers to test pure character-level staggered temporal networks.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)
