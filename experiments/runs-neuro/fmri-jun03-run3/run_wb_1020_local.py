from src.eval import run_encoding, EncodingConfig, make_result_row, upsert_overall_results
import torch
import torch.nn as nn
from interpretable_transformer import SimpleTransformer, VOCAB, InterpretableEmbedder
import math

def build_wb_1020():
    d_model = 1020
    d_ff = 4000
    n_heads = 10
    n_layers = 2
    max_seq_len = 64
    
    model = SimpleTransformer(
        vocab_size=len(VOCAB), max_seq_len=max_seq_len,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
        
    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.zeros_(p)
    
    with torch.no_grad():
        vocab_size = model.vocab_size

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
            pos_emb[p, 29] = 1.0 # Bias for attention
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        l1_attn.W_q.weight[0, 29] = 5.0
        l1_attn.W_k.weight[0, 28] = 1.0
        
        for i in range(28):
            l1_attn.W_v.weight[i, i] = 1.0
            l1_attn.W_o.weight[30 + i, i] = 1.0
            
        mlp1 = model.blocks[0].mlp
        nn.init.normal_(mlp1.fc1.weight, std=0.5)
        nn.init.normal_(mlp1.fc2.weight, std=0.5)
        
        l2_attn = model.blocks[1].attn
        for i in range(d_model):
            l2_attn.W_q.weight[i, i] = 1.0
            l2_attn.W_k.weight[i, i] = 1.0
            l2_attn.W_v.weight[i, i] = 1.0
            l2_attn.W_o.weight[i, i] = 1.0
            
        mlp2 = model.blocks[1].mlp
        nn.init.normal_(mlp2.fc1.weight, std=0.5)
        nn.init.normal_(mlp2.fc2.weight, std=0.5)

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.ones_(block.ln2.weight)
                
        nn.init.ones_(model.final_ln.weight)
        
    model.eval()
    return InterpretableEmbedder(model, device='cuda')

embedder = build_wb_1020()
cfg = EncodingConfig(subject="UTS03", num_train=8, num_test=2)
r = run_encoding(embedder, cfg)
n_params = sum(p.numel() for p in embedder.model.parameters())

print(f"train_corr={r['corrs_train_mean']:.4f} test_corr={r['test_corr']:.4f}")

upsert_overall_results(
    [make_result_row(r, "ExactWordBoundary1020", n_params, "Exact logic but scaled to 1020")], 
    "results")
