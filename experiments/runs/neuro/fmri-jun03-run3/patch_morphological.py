import os
import sys
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# 1. Update VOCAB
old_vocab = """_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\\\\\""
VOCAB = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)"""

new_vocab = """_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\\\\\""
MORPH_SUFFIXES = ["ing", "ed", "ly", "es", "tion", "ment", "ness", "ity", "er", "est", "ful", "able", "ive", "less", "ize", "ate"]
VOCAB = ['<pad>', '<unk>'] + list(_VOCAB_CHARS) + MORPH_SUFFIXES
"""
content = content.replace(old_vocab, new_vocab)


# 2. Update InterpretableEmbedder
old_embedder = """    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, s in enumerate(input_strings):
            for j, char in enumerate(s[-T:]):
                if char in VOCAB:
                    input_ids[i, j] = VOCAB.index(char)
                else:
                    input_ids[i, j] = VOCAB.index('<unk>')
        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()"""

new_embedder = """    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        
        # Sort vocab by length descending for greedy matching
        sorted_vocab = sorted(VOCAB[2:], key=len, reverse=True)
        
        for i, s in enumerate(input_strings):
            s = s[-T*4:] # get enough chars
            tokens = []
            idx = 0
            while idx < len(s):
                matched = False
                for v in sorted_vocab:
                    if s[idx:].startswith(v):
                        tokens.append(v)
                        idx += len(v)
                        matched = True
                        break
                if not matched:
                    tokens.append('<unk>')
                    idx += 1
            
            tokens = tokens[-T:]
            padded_tokens = ['<pad>'] * (T - len(tokens)) + tokens
            
            for j, t in enumerate(padded_tokens):
                if t in VOCAB:
                    input_ids[i, j] = VOCAB.index(t)
                else:
                    input_ids[i, j] = VOCAB.index('<unk>')
                    
        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()"""

content = content.replace(old_embedder, new_embedder)

# 3. Update write_weights to inject Suffixes into dims 29-45
old_inject = """        for i, c in enumerate(VOCAB):
            if c == '<pad>' or c == '<unk>':
                continue
            elif len(c) == 1 and c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0
            else:
                token_emb[i, 0] = S * 1.0

        for net_idx in range(num_nets):
            start_dim = net_idx * dim_per_net
            # Inject character feature
            token_emb[:, start_dim:start_dim+28] = token_emb[:, 0:28]"""

new_inject = """        for i, c in enumerate(VOCAB):
            if c == '<pad>' or c == '<unk>':
                continue
            elif len(c) == 1 and c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0
            elif c in MORPH_SUFFIXES:
                # Store suffix identity in 29-45
                suffix_idx = MORPH_SUFFIXES.index(c)
                token_emb[i, 29 + suffix_idx] = S * 1.0
            else:
                token_emb[i, 0] = S * 1.0

        for net_idx in range(num_nets):
            start_dim = net_idx * dim_per_net
            # Inject character feature AND morphological features
            token_emb[:, start_dim:start_dim+46] = token_emb[:, 0:46]"""
            
content = content.replace(old_inject, new_inject)

content = content.replace("Deep_Ensemble_0421_Master", "Morphological_Subword_Tracker")
content = content.replace("Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures.", "Adds a greedy subword tokenizer to extract explicit morphological suffixes (ing, ed, ly, tion) and temporally smears them alongside the characters to capture syntactic tense and POS structure.")

with open(filepath, "w") as f:
    f.write(content)

print("Generated Morphological Subword Tracker")
