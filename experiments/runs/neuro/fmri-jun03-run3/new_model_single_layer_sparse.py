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

        # We know PerfectSparseWordBoundary (which is 1 network of 64 dims) gets 0.0355
        # We know PerfectEnsembleWordBoundary (which is 15 networks of 64 dims) gets 0.0355
        # The ensemble didn't add anything over the single sparse network!
        
        # Why did WordBoundaryFeatures get 0.0405 when we evaluated it earlier?
        # ExactWordBoundaryRepro (where we forced d_model=64) got 0.0328 in a recent test!
        # Wait... the original WordBoundaryFeatures scored 0.0405.
        # But when we ran ExactWordBoundaryRepro we only got 0.0328.
        # Let's look at the ExactWordBoundaryRepro code compared to original WordBoundaryFeatures.

        # The only difference is the initialization seed! 
        # Deep learning models with random features are sensitive to the random seed.
        # If the seed was different, the 0.0405 might have just been a lucky seed.
        # Or... maybe we can just expand the MLP width to give it more "chances" to find good features,
        # but keep the projection dimension (d_model) small.

        # Let's build a single sparse network (d_model=64), but let's use the FULL d_ff=4000 capacity!
        # This will be a single network with a massive hidden layer, but small bottleneck.

        num_nets = 1
        dim_per_net = 64
        ff_per_net = d_ff # All 4000
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        d_start = 0
        f_start = 0
        h_start = 0
        
        l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
        l1_attn.W_k.weight[h_start + 0, d_start + 28] = 1.0
        
        for i in range(28):
            l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.0
            
        # Massive MLP Expansion!
        nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=0.5)
        nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=0.5 * S)
        
        for i in range(dim_per_net):
            l2_attn.W_q.weight[h_start + i, d_start + i] = 1.0
            l2_attn.W_k.weight[h_start + i, d_start + i] = 1.0
            l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
            l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
            
        nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=0.5)
        nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=0.5 * S)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "SparseWordBoundary_WideMLP"
model_description = "Uses the variance anchor trick to create a single 64-dim network, but allocates the entire 4000-dim hidden layer capacity to it to maximize random feature coverage."
