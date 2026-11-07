import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformers_lib/Final_Interpretable_Absolute_SOTA.py", "r") as f:
    content = f.read()

pattern = r"std_dev = 0\.5.*?std_dev \* S\)"
replacement = "pass"
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

pattern2 = r"std_dev2 = 0\.01 \+ float\(net\) \* \(3\.99 / 14\.0\).*?std_dev2 \* S\)"
replacement2 = "pass"
content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

content = content.replace('model_shorthand_name = "Interpretable_Staggered_Absolute_Morph_Peak"', 'model_shorthand_name = "Interpretable_Linear_Attention_Only"')
content = content.replace('model_description = "The absolute ceiling', 'model_description = "Complete removal of MLPs (except morphology). Testing pure linear attention integration.')

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)
