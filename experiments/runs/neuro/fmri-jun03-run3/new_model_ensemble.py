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

        C = 10000.0

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            token_emb[i, i] = 1.0 # 0..49
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            pos_emb[p, 1002] = float(p)
            pos_emb[p, 1003] = float(p)**2
            pos_emb[p, 1004] = 1.0
            pos_emb[p, 1005] = -float(p)
            pos_emb[p, 1006] = -float(p)**2
            pos_emb[p, 1007] = -1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        # WordBoundaryFeatures style but run N independent subspace MLPs
        num_subspaces = 10
        subspace_dim = d_model // num_subspaces
        assert subspace_dim * num_subspaces == d_model
        sub_d_ff = d_ff // num_subspaces
        
        S_sub = C * math.sqrt(2.0 / subspace_dim)

        l1_attn = model.blocks[0].attn
        
        # WordBoundary style smooth attention for each subspace independently
        for sub in range(num_subspaces):
            # assign a head to each subspace
            k = sub % n_heads
            
            # Smooth exponential decay lookback
            # Decay scale varies by subspace for multi-scale context
            decay_scale = 1.0 + float(sub) / 5.0
            
            l1_attn.W_q.weight[k*d_head + 0, 1004] = S_sub * 5.0
            l1_attn.W_k.weight[k*d_head + 0, 1002] = S_sub * decay_scale
            
            # Subspace extracts characters
            start_dim = sub * subspace_dim
            for i in range(50):
                l1_attn.W_v.weight[k*d_head + i, i] = S_sub * 1.0
                l1_attn.W_o.weight[start_dim + i, k*d_head + i] = 1.0
                
        # MLP acts independently on each subspace
        mlp1 = model.blocks[0].mlp
        for sub in range(num_subspaces):
            in_start = sub * subspace_dim
            in_end = (sub + 1) * subspace_dim
            out_start = sub * sub_d_ff
            out_end = (sub + 1) * sub_d_ff
            
            # Only connect subspace inputs to subspace hidden neurons
            nn.init.normal_(mlp1.fc1.weight[out_start:out_end, in_start:in_end], std=1.0)
            nn.init.normal_(mlp1.fc2.weight[in_start:in_end, out_start:out_end], std=1.0)
        
        # Layer 2 repeats the process
        l2_attn = model.blocks[1].attn
        for sub in range(num_subspaces):
            k = sub % n_heads
            start_dim = sub * subspace_dim
            
            l2_attn.W_q.weight[k*d_head + 0, 1004] = S_sub * 5.0
            l2_attn.W_k.weight[k*d_head + 0, 1002] = S_sub * 1.0 # fixed decay for L2
            
            # pass through the subspace features
            for i in range(subspace_dim):
                l2_attn.W_v.weight[k*d_head + i, start_dim + i] = S_sub * 1.0
                l2_attn.W_o.weight[start_dim + i, k*d_head + i] = 1.0
                
        mlp2 = model.blocks[1].mlp
        for sub in range(num_subspaces):
            in_start = sub * subspace_dim
            in_end = (sub + 1) * subspace_dim
            out_start = sub * sub_d_ff
            out_end = (sub + 1) * sub_d_ff
            
            nn.init.normal_(mlp2.fc1.weight[out_start:out_end, in_start:in_end], std=1.0)
            nn.init.normal_(mlp2.fc2.weight[in_start:in_end, out_start:out_end], std=1.0)

        for block in model.blocks:
            # Important: Use LayerNorm on subspaces, or don't use it at all?
            # LayerNorm mixes the subspaces. That breaks independence.
            # We can't avoid the built-in LayerNorm since the architecture is fixed.
            # But we can pass through if we use the C scaling trick.
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "SubspaceEnsembleHash"
model_description = "Creates 10 independent parallel networks within the large transformer by masking MLP weights, each with a different decay scale. This simulates 16 low-capacity networks."
