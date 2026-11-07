import re

with open("interpretable_transformer.py", "r") as f:
    content = f.read()

with open("interpretable_transformers_lib/WordBoundaryFeatures.py", "r") as f:
    orig_code = f.read()

# We need to extract write_weights from orig_code
start_idx = orig_code.find("def write_weights")
end_idx = orig_code.find("model_shorthand_name = \"WordBoundaryFeatures\"")

code_to_insert = orig_code[start_idx:end_idx] + """
model_shorthand_name = "WordBoundaryFeaturesReal"
model_description = "Hand-designed orthographic features (is_space, letter identity) + recent context + strong random MLPs to detect word-level patterns."
"""

target_start_idx = content.find("def write_weights")
target_end_idx = content.find("def build_embedder")

new_content = content[:target_start_idx] + code_to_insert + "\n\n" + content[target_end_idx:]

with open("interpretable_transformer.py", "w") as f:
    f.write(new_content)
