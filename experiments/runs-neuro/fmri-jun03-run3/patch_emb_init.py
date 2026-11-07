import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Make the initial token embeddings much richer/denser instead of just binary alphabet indicators
old_emb = """        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        # Write exactly to the last 28 dimensions
        token_emb[:, 960:988] = token_emb[:, 0:28]"""

new_emb = """        token_emb = torch.zeros(vocab_size, d_model)
        
        # Add normal noise as a dense baseline semantic embedding
        nn.init.normal_(token_emb, std=S * 0.1)
        
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 
                
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        # Write exactly to the last 28 dimensions
        token_emb[:, 960:988] = token_emb[:, 0:28]"""

content = content.replace(old_emb, new_emb)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Dense_Emb")

with open(filepath, "w") as f:
    f.write(content)
print("Applied dense embedding patch")
