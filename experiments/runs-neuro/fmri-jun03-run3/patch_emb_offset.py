import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Add a dense representation where characters activate multiple adjacent alphabet indicators
old_emb = """        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 """

new_emb = """        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                idx = ord(c) - ord('a')
                token_emb[i, 2 + idx] = S * 1.0
                # Add adjacent letters with slight blur
                if idx > 0:
                    token_emb[i, 2 + idx - 1] = S * 0.2
                if idx < 25:
                    token_emb[i, 2 + idx + 1] = S * 0.2"""

content = content.replace(old_emb, new_emb)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Blurred_Emb")

with open(filepath, "w") as f:
    f.write(content)
print("Applied blurred embedding patch")
