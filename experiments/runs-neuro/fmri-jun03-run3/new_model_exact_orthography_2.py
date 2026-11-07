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

        # All of our variations (wider, n-gram, varying standard deviation) 
        # score ~0.033-0.036.
        # This points to a ceiling on random orthographic mixing.
        # Let's try explicit hard-coded features for the first time, mixed with
        # a standard ensemble for the remaining capacity.
        # Network 1: Word length counter
        # Network 2: Space detector
        # Network 3: Vowel counter
        # Network 4: Consonant counter
        # Networks 5-15: Standard decaying sparse ensemble.
        
        num_nets = 15
        dim_per_net = 64
        ff_per_net = 256
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 # Base space detector
            elif c.isalpha():
                token_emb[i, 1] = S * 1.0
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0
                
                # Setup vowels/consonants
                if c in 'aeiou':
                    token_emb[i, 30] = S * 1.0
                else:
                    token_emb[i, 31] = S * 1.0
                    
        for net in range(num_nets):
            start = net * dim_per_net
            token_emb[:, start:start+32] = token_emb[:, 0:32]
            
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1000] = C
            pos_emb[p, 1001] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
            
        for net in range(num_nets):
            start = net * dim_per_net
            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]
            
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            if net == 0:
                # Word length counter - no decay, sums across word
                # Attention ignores spaces
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 0] = -5.0 # Attn drops sharply on space
                
                l1_attn.W_v.weight[h_start + 0, d_start + 1] = 1.0 # Value is character existence
                l1_attn.W_o.weight[d_start + 32, h_start + 0] = S * 1.0 # Output word length
                
            elif net == 1:
                # Space detector - pure decay from last space
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 0] = 5.0 # Attend strongly to space
                
                l1_attn.W_v.weight[h_start + 0, d_start + 0] = 1.0 # Value is space existence
                l1_attn.W_o.weight[d_start + 33, h_start + 0] = S * 1.0
                
            elif net == 2:
                # Vowel counter
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 28] = 2.0
                l1_attn.W_v.weight[h_start + 0, d_start + 30] = 1.0
                l1_attn.W_o.weight[d_start + 34, h_start + 0] = S * 1.0
                
            elif net == 3:
                # Consonant counter
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 28] = 2.0
                l1_attn.W_v.weight[h_start + 0, d_start + 31] = 1.0
                l1_attn.W_o.weight[d_start + 35, h_start + 0] = S * 1.0
                
            else:
                # Standard ensemble behavior for the rest
                decay_scale = 1.0 + float(net - 4) / 4.0
                
                l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
                l1_attn.W_k.weight[h_start + 0, d_start + 28] = decay_scale
                
                for i in range(28):
                    l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                    l1_attn.W_o.weight[d_start + 36 + i, h_start + i] = S * 1.0
                
            std_dev = 1.0
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            for i in range(dim_per_net):
                l2_attn.W_q.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_k.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0
                
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)

model_shorthand_name = "Explicit_Extractors"
model_description = "Uses specific networks for word length, space detection, vowels, and consonants, alongside a standard sparse ensemble."
