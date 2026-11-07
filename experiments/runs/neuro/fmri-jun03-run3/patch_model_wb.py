import re

with open("interpretable_transformer.py", "r") as f:
    content = f.read()

with open("new_model_word_boundary.py", "r") as f:
    new_model_code = f.read()

start_idx = content.find("def write_weights")
end_idx = content.find("def build_embedder")

new_start_idx = new_model_code.find("def write_weights")
new_code_to_insert = new_model_code[new_start_idx:]

new_content = content[:start_idx] + new_code_to_insert + "\n" + content[end_idx:]

with open("interpretable_transformer.py", "w") as f:
    f.write(new_content)
