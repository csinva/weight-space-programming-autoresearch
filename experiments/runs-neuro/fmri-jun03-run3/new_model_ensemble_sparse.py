import math
import torch
import torch.nn as nn
from interpretable_transformer import SimpleTransformer, VOCAB

def write_weights(model: SimpleTransformer) -> None:
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        d_model = model.d_model
        max_seq_len = model.max_seq_len
        vocab_size = model.vocab_size
        d_ff = model.blocks[0].mlp.fc1.out_features
        n_heads = model.blocks[0].attn.n_heads
        d_head = model.blocks[0].attn.d_head

        # Ensembles of EXTREMELY SMALL capacity
        # The best model was d_model=64, test=0.0405. 
        # But wait! Scaled down to 1020, it scores 0.0247.
        # This confirms my theory: WordBoundaryFeatures ONLY works because of its low capacity regularizing it.
        # But my subspace ensemble scored 0.0304.
        
        # Let's try simulating the WordBoundaryFeatures with d_model=64 inside the 1024 model
        # but just doing it ONCE, and leaving the rest of the 1024 dimensions EMPTY.
        # This should theoretically get 0.0405 again, proving it.

        C = 10000.0

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = 1.0
            elif c.isalpha():
                token_emb[i, 1] = 1.0
                token_emb[i, 2 + (ord(c) - ord('a'))] = 1.0
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 28] = p / max_seq_len
            pos_emb[p, 29] = 1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        
        l1_attn.W_q.weight[0, 29] = 5.0
        l1_attn.W_k.weight[0, 28] = 1.0
        
        for i in range(28):
            l1_attn.W_v.weight[i, i] = 1.0
            l1_attn.W_o.weight[30 + i, i] = 1.0
            
        mlp1 = model.blocks[0].mlp
        nn.init.normal_(mlp1.fc1.weight[:256, :64], std=0.5)
        nn.init.normal_(mlp1.fc2.weight[:64, :256], std=0.5)
        
        l2_attn = model.blocks[1].attn
        for i in range(64):
            l2_attn.W_q.weight[i, i] = 1.0
            l2_attn.W_k.weight[i, i] = 1.0
            l2_attn.W_v.weight[i, i] = 1.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight[:256, :64], std=0.5)
        nn.init.normal_(mlp2.fc2.weight[:64, :256], std=0.5)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight[:64])
            nn.init.ones_(block.ln2.weight[:64])
                
        nn.init.ones_(model.final_ln.weight[:64])

model_shorthand_name = "SparseWordBoundary"
model_description = "Creates exactly one instance of the 64-dim WordBoundaryFeatures, leaving the other 956 dimensions completely zeroed out, to prove the capacity constraint."
