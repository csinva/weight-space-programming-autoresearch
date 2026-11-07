import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

weight_replacement = """
        token_emb = torch.zeros(vocab_size, d_model)
        for i, w in enumerate(VOCAB):
            if w == '<pad>':
                continue
            
            # Everyone gets the base word feature
            token_emb[i, 1] = S * 1.0
            
            # Words with specific characters
            if 'a' in w: token_emb[i, 2] = S * 1.0
            if 'e' in w: token_emb[i, 3] = S * 1.0
            if 'i' in w: token_emb[i, 4] = S * 1.0
            if 'o' in w: token_emb[i, 5] = S * 1.0
            if 'u' in w: token_emb[i, 6] = S * 1.0
            
            # Word length features
            length = len(w)
            if length <= 3: token_emb[i, 7] = S * 1.0
            elif length <= 5: token_emb[i, 8] = S * 1.0
            else: token_emb[i, 9] = S * 1.0
            
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        # We can also add random hashes for top words to create exact matches
        np.random.seed(42)
        dense_hash = np.random.randn(vocab_size, 28) * S
        token_emb[:, 960:988] = torch.tensor(dense_hash, dtype=torch.float32)
"""

content = re.sub(
    r"        token_emb = torch\.zeros\(vocab_size, d_model\)\n        for i, c in enumerate\(VOCAB\):.*?\n        token_emb\[:, 960:988\] = token_emb\[:, 0:28\]",
    weight_replacement.strip(),
    content,
    flags=re.DOTALL
)

# Update model description
content = content.replace("model_shorthand_name = \"Deep_Ensemble_0421_Master\"", "model_shorthand_name = \"Word_Level_Ensemble\"")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "Transforms the entire architecture from Character-Level to Word-Level using TOP_WORDS[:2000] and maps the stagger delays across word boundaries.")

# Set sequence length to 10 words
content = content.replace("max_seq_len: int = 64", "max_seq_len: int = 10")


with open(filepath, "w") as f:
    f.write(content)
