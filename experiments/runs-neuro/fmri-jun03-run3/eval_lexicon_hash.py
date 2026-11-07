import math
import torch
import torch.nn as nn
from interpretable_transformer import SimpleTransformer, VOCAB
from top_words import TOP_WORDS

def write_weights(model: SimpleTransformer) -> None:
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        d_model = model.d_model
        max_seq_len = model.max_seq_len
        vocab_size = model.vocab_size
        n_heads = model.blocks[0].attn.n_heads
        d_head = model.blocks[0].attn.d_head
        d_ff = model.blocks[0].mlp.fc1.out_features

        C = 10000.0
        S = C / math.sqrt(d_model)

        # We will use the first 10 heads for 10 different decay rates
        num_decays = 10
        decays = [10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
        
        # Token embeddings: Just one-hot characters in the first 28 dims
        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S
            elif c.isalpha():
                token_emb[i, 1 + (ord(c) - ord('a'))] = S
            else:
                token_emb[i, 27] = S
        model.token_emb.weight.data.copy_(token_emb)

        # Positional embeddings: ramp for exponential decay
        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 28] = S * (p / max_seq_len)
        model.pos_emb.weight.data.copy_(pos_emb)

        # Layer 1 Attention: Compute continuous EMA for characters
        l1_attn = model.blocks[0].attn
        for h in range(num_decays):
            h_start = h * d_head
            decay = decays[h]
            l1_attn.W_q.weight[h_start + 0, 1000] = 5.0
            l1_attn.W_k.weight[h_start + 0, 28] = decay
            
            # Pass the 28 char dims through
            for i in range(28):
                l1_attn.W_v.weight[h_start + i, i] = 1.0
                l1_attn.W_o.weight[h * 28 + i, h_start + i] = S

        # Precompute signatures for TOP_WORDS
        # We take the top 500 words
        target_words = TOP_WORDS[:500]
        
        # Signature dimension = num_decays * 28 = 280
        signatures = torch.zeros(len(target_words), 280)
        
        for w_idx, w in enumerate(target_words):
            # The sequence evaluated is "... space + word"
            # evaluated at the LAST character of the word.
            seq = " " + w
            for j in range(len(seq)):
                c = seq[len(seq) - 1 - j]
                if c == ' ':
                    c_idx = 0
                elif c.isalpha():
                    c_idx = 1 + (ord(c) - ord('a'))
                else:
                    c_idx = 27
                    
                for h in range(num_decays):
                    val = math.exp(-decays[h] * j)
                    signatures[w_idx, h * 28 + c_idx] += val
                    
        # Normalize signatures
        norms = torch.norm(signatures, dim=1, keepdim=True)
        signatures = signatures / (norms + 1e-8)

        # Layer 1 MLP: Template matching
        mlp1 = model.blocks[0].mlp
        # fc1 takes the 280 dims, outputs len(target_words)
        for w_idx in range(len(target_words)):
            mlp1.fc1.weight[w_idx, 0:280] = signatures[w_idx] * 2.0  # Scale up for ReLU
            mlp1.fc1.bias[w_idx] = -1.8  # Threshold
            
            # Map detection to a dedicated dimension in d_model
            # Dims 300 to 800 will be one-hot word indicators
            mlp1.fc2.weight[300 + w_idx, w_idx] = S

        # Layer 2 Attention: Temporal Integration (BOLD Hemodynamic Smoothing)
        # We use a broad stagger across the word indicators
        l2_attn = model.blocks[1].attn
        
        # 10 heads for temporal integration of the word vectors
        # Let's use decays from 0.01 to 1.0
        l2_decays = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]
        for h in range(10):
            h_start = h * d_head
            decay = l2_decays[h]
            l2_attn.W_q.weight[h_start + 0, 1000] = 5.0
            l2_attn.W_k.weight[h_start + 0, 28] = decay
            
            # Pass through 50 words per head (50 * 10 = 500)
            for i in range(50):
                w_idx = h * 50 + i
                l2_attn.W_v.weight[h_start + 1 + i, 300 + w_idx] = 1.0
                l2_attn.W_o.weight[300 + w_idx, h_start + 1 + i] = S

        # Layer 2 MLP: Just pass it through
        # Actually skip connection already passes it through, so we can leave MLP2 zero.
        
