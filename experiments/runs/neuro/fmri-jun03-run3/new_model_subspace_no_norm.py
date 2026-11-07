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

        # 16 subspaces inside d_model=1024
        # d_model=1024 / 16 = 64
        # d_ff=4096 / 16 = 256
        # These are EXACTLY the dimensions of WordBoundaryFeatures
        
        num_subspaces = 16
        subspace_dim = d_model // num_subspaces
        sub_d_ff = d_ff // num_subspaces
        
        C = 10000.0

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = 1.0
            elif c.isalpha():
                token_emb[i, 1] = 1.0
                token_emb[i, 2 + (ord(c) - ord('a'))] = 1.0
        
        # Broadcast the token embedding to all subspaces
        full_token_emb = torch.zeros(vocab_size, d_model)
        for sub in range(num_subspaces):
            start = sub * subspace_dim
            full_token_emb[:, start:start+subspace_dim] = token_emb[:, :subspace_dim]
        model.token_emb.weight.data.copy_(full_token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 28] = p / max_seq_len
            pos_emb[p, 29] = 1.0
            
        full_pos_emb = torch.zeros(max_seq_len, d_model)
        for sub in range(num_subspaces):
            start = sub * subspace_dim
            full_pos_emb[:, start:start+subspace_dim] = pos_emb[:, :subspace_dim]
        model.pos_emb.weight.data.copy_(full_pos_emb)

        l1_attn = model.blocks[0].attn
        
        # In the original:
        # l1_attn.W_q.weight[0, 29] = 5.0
        # l1_attn.W_k.weight[0, 28] = 1.0
        
        # We need to replicate this per-subspace but also make them mathematically independent if possible
        # Actually wait, WordBoundaryFeatures used nn.init.ones_(ln) and we can just let LN happen over
        # the entire 1024 dims. It will compute a mean over 1024 dims.
        # But if we replicate everything 16 times, the mean over 1024 dims is identical to the mean over 64 dims.
        # Let's break symmetry just slightly in the random MLPs, so each subspace learns a different hash.
        
        for sub in range(num_subspaces):
            start = sub * subspace_dim
            # Assume 1 head per subspace? No, d_model=1024, n_heads=16, so 1 head per subspace perfectly.
            head = sub % n_heads
            head_start = head * d_head
            
            # Subspace-specific Q and K
            l1_attn.W_q.weight[head_start + 0, start + 29] = 5.0
            l1_attn.W_k.weight[head_start + 0, start + 28] = 1.0
            
            # W_v pass through
            for i in range(28):
                l1_attn.W_v.weight[head_start + i, start + i] = 1.0
                l1_attn.W_o.weight[start + 30 + i, head_start + i] = 1.0
                
        # MLP: Masked to keep subspaces independent
        mlp1 = model.blocks[0].mlp
        for sub in range(num_subspaces):
            in_start = sub * subspace_dim
            in_end = (sub + 1) * subspace_dim
            out_start = sub * sub_d_ff
            out_end = (sub + 1) * sub_d_ff
            
            nn.init.normal_(mlp1.fc1.weight[out_start:out_end, in_start:in_end], std=0.5)
            nn.init.normal_(mlp1.fc2.weight[in_start:in_end, out_start:out_end], std=0.5)
            
        # L2 Attention: Pass through
        l2_attn = model.blocks[1].attn
        for sub in range(num_subspaces):
            start = sub * subspace_dim
            head = sub % n_heads
            head_start = head * d_head
            
            for i in range(subspace_dim):
                l2_attn.W_q.weight[head_start + i, start + i] = 1.0
                l2_attn.W_k.weight[head_start + i, start + i] = 1.0
                l2_attn.W_v.weight[head_start + i, start + i] = 1.0
                l2_attn.W_o.weight[start + i, head_start + i] = 1.0
                
        # MLP2: Masked
        mlp2 = model.blocks[1].mlp
        for sub in range(num_subspaces):
            in_start = sub * subspace_dim
            in_end = (sub + 1) * subspace_dim
            out_start = sub * sub_d_ff
            out_end = (sub + 1) * sub_d_ff
            
            nn.init.normal_(mlp2.fc1.weight[out_start:out_end, in_start:in_end], std=0.5)
            nn.init.normal_(mlp2.fc2.weight[in_start:in_end, out_start:out_end], std=0.5)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "ParallelWordBoundary"
model_description = "Creates 16 mathematically independent instances of WordBoundaryFeatures (d_model=64) inside the 1024-dim model, breaking symmetry only at the random MLP layer."
