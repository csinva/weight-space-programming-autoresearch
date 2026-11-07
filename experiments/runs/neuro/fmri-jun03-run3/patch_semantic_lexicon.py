import re

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "r") as f:
    content = f.read()

# Modify InterpretableEmbedder to force a semantic lexicon lookup
# We'll use a simple hashing trick to group characters into "words" based on spaces
# and force the representation to be a bag-of-words or semantic lexicon hash.

emb_old = """        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()"""

emb_new = """        hidden_states = self.model(input_ids)
        
        # Semantic Lexicon Bottleneck Hypothesis
        # Brains comprehend language through discrete semantic concepts (words/morphemes),
        # not just continuous acoustic/orthographic strings.
        # We enforce a semantic bottleneck: we zero out the continuous representation 
        # and replace it with a hashed Lexicon Vector derived purely from the last complete word.
        
        final_state = hidden_states[:, -1, :].clone()
        B, T = input_ids.shape
        D = final_state.shape[1]
        
        # We don't have access to the strings here easily, but we have input_ids
        # space is VOCAB.index(' ') = 0
        for i in range(B):
            # find last space
            ids = input_ids[i].tolist()
            last_space = T - 1
            for j in range(T - 1, -1, -1):
                if ids[j] == 0:
                    last_space = j
                    break
                    
            # Extract the last word
            word_ids = tuple(ids[last_space+1:])
            # Hash the word to a specific representation vector
            _hash = hash(word_ids) % 1000000
            
            # Use the hash as a seed to generate a consistent random semantic vector for this word
            torch.manual_seed(_hash)
            semantic_vector = torch.randn(D, device=hidden_states.device)
            
            # Override the continuous state with the discrete semantic lexicon state
            final_state[i] = semantic_vector
            
        return final_state.cpu()"""

content = content.replace(emb_old, emb_new)

# Replace description
desc_old = 'model_shorthand_name = "Deep_Ensemble_0421_Master"'
desc_new = 'model_shorthand_name = "Semantic_Lexicon_Bottleneck"'
content = content.replace(desc_old, desc_new)

desc_old2 = 'model_description = "Uses the exact optimal scales of UltraTune, but changes the staggering from a 3-way split (+0, +6, +12) with L1 decay scale set to 15-80 instead of 10-80 to create even richer timescale mixtures."'
desc_new2 = 'model_description = "Semantic Lexicon Bottleneck Hypothesis: Human comprehension ultimately relies on mapping acoustic/orthographic streams into discrete semantic concepts (words in a mental lexicon). Tested this by completely wiping the continuous character-level hidden state and replacing it with a static, hashed Semantic Vector representing strictly the last perceived word (acting as a pure Bag-of-Words lexicon lookup)."'
content = content.replace(desc_old2, desc_new2)

with open("runs-neuro/fmri-jun03-run3/interpretable_transformer.py", "w") as f:
    f.write(content)

print("Patched Semantic Lexicon Bottleneck.")
