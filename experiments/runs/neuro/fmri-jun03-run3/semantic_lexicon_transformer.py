"""
Interpretable transformer embedder for fMRI language encoding.
"""

from __future__ import annotations

import math
import argparse
import os
import sys
import time
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

SEMANTIC_WORDS: Dict[str, set[str]] = {
    "first_person": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"},
    "second_person": {"you", "your", "yours", "yourself", "yourselves"},
    "third_person": {"he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself"},
    "plural_pronoun": {"we", "us", "our", "they", "them", "their", "theirs", "everyone", "somebody"},
    "social_family": {"mother", "mom", "mommy", "father", "dad", "daddy", "parent", "parents", "brother", "sister", "son", "daughter", "child", "children", "kid", "kids", "wife", "husband", "boyfriend", "girlfriend", "friend", "friends", "family", "grandmother", "grandfather", "grandma", "grandpa"},
    "people": {"man", "woman", "men", "women", "boy", "girl", "person", "people", "guy", "lady", "child", "children", "kid", "teacher", "doctor", "student", "police", "cop", "stranger", "neighbor", "boss", "worker", "baby"},
    "communication": {"say", "said", "says", "tell", "told", "talk", "talked", "speak", "spoke", "ask", "asked", "answer", "answered", "call", "called", "shout", "shouted", "whisper", "whispered", "voice", "word", "words", "story", "read", "write", "heard", "listen", "conversation"},
    "cognition": {"think", "thought", "know", "knew", "believe", "remember", "forgot", "forget", "understand", "wonder", "guess", "decide", "decided", "realize", "realized", "idea", "mind", "dream", "imagine", "learn", "learned", "mean", "meant"},
    "emotion_positive": {"love", "loved", "like", "liked", "happy", "glad", "laugh", "laughed", "smile", "smiled", "fun", "funny", "nice", "good", "great", "beautiful", "safe", "hope"},
    "emotion_negative": {"hate", "hated", "sad", "angry", "mad", "afraid", "scared", "fear", "worried", "worry", "cry", "cried", "hurt", "pain", "bad", "wrong", "terrible", "dead", "death", "kill", "killed", "alone", "sorry"},
    "motion": {"go", "goes", "went", "gone", "come", "came", "walk", "walked", "run", "ran", "move", "moved", "turn", "turned", "stand", "stood", "sit", "sat", "leave", "left", "enter", "entered", "drive", "drove", "fall", "fell", "jump", "climb"},
    "perception": {"see", "saw", "seen", "look", "looked", "watch", "watched", "hear", "heard", "listen", "listened", "feel", "felt", "smell", "taste", "notice", "noticed"},
    "body": {"hand", "hands", "arm", "arms", "leg", "legs", "head", "face", "eye", "eyes", "mouth", "hair", "heart", "body", "back", "feet", "foot", "finger", "fingers", "skin", "blood", "brain"},
    "place_scene": {"room", "house", "home", "street", "road", "car", "school", "city", "town", "store", "office", "door", "window", "bed", "table", "kitchen", "bathroom", "park", "river", "water", "church", "hospital", "apartment"},
    "time": {"time", "day", "night", "morning", "evening", "hour", "minute", "second", "week", "month", "year", "today", "yesterday", "tomorrow", "then", "now", "before", "after", "later", "again", "always", "never"},
    "number_quantity": {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first", "second", "third", "many", "much", "few", "more", "less", "all", "some", "any", "each", "every", "both", "half"},
    "object_artifact": {"thing", "things", "book", "paper", "phone", "letter", "money", "gun", "knife", "bag", "box", "chair", "table", "bed", "clothes", "shirt", "shoes", "picture", "photo", "key", "keys", "glass", "bottle", "computer"},
    "food_drink": {"food", "eat", "ate", "eating", "drink", "drank", "water", "coffee", "tea", "beer", "wine", "bread", "milk", "meat", "cake", "dinner", "breakfast", "lunch"},
    "animal_nature": {"dog", "cat", "bird", "horse", "animal", "tree", "trees", "flower", "flowers", "sun", "moon", "sky", "rain", "snow", "wind", "fire", "earth", "forest", "sea"},
    "abstract_logic": {"because", "if", "though", "although", "maybe", "probably", "perhaps", "why", "reason", "cause", "truth", "true", "false", "fact", "problem", "question"},
    "negation": {"no", "not", "never", "nothing", "nobody", "none", "cannot", "cant", "don't", "didn't"},
    "question": {"who", "what", "when", "where", "why", "how", "which", "question", "ask", "asked"},
    "determiner": {"the", "a", "an", "this", "that", "these", "those", "each", "every"},
    "preposition": {"in", "on", "at", "by", "for", "with", "without", "from", "to", "into", "onto", "over", "under", "between", "through", "around", "about", "against", "inside", "outside", "before", "after"},
    "conjunction": {"and", "or", "but", "so", "because", "while", "though", "although", "if", "then"},
    "auxiliary": {"is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "should", "may", "might", "must"},
}
SEMANTIC_KEYS = list(SEMANTIC_WORDS.keys())

VOCAB_WORDS = []
for words in SEMANTIC_WORDS.values():
    VOCAB_WORDS.extend(list(words))
VOCAB_WORDS = list(set(VOCAB_WORDS))

_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
VOCAB = ['<pad>', '<unk>'] + list(_VOCAB_CHARS) + VOCAB_WORDS

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int,
                 n_heads: int, n_layers: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return x

class InterpretableEmbedder(nn.Module):
    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        super().__init__()
        self.model = model.to(device)
        self.device = device

    def encode_text(self, text: str) -> torch.Tensor:
        text = text[-self.model.max_seq_len:]
        text = text.ljust(self.model.max_seq_len, ' ')
        
        ids = [VOCAB.index(c) if c in VOCAB else VOCAB.index('<unk>') for c in text]
        
        import re
        words = re.findall(r"[\w']+|[.,!?;]", text.lower())
        
        char_idx = 0
        for w in words:
            start_idx = text.lower().find(w, char_idx)
            if start_idx != -1:
                end_idx = start_idx + len(w) - 1
                if w in VOCAB:
                    ids[end_idx] = VOCAB.index(w)
                char_idx = end_idx + 1
        
        return torch.tensor(ids, dtype=torch.long)

    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, s in enumerate(input_strings):
            input_ids[i] = self.encode_text(s)
        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()

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

        num_nets = 16
        dim_per_net = 64
        ff_per_net = 256
        
        C = 10000.0
        S = C / math.sqrt(d_model)

        token_emb = torch.zeros(vocab_size, d_model)
        for i, c in enumerate(VOCAB):
            if c == ' ':
                token_emb[i, 0] = S * 1.0 
            elif len(c) == 1 and c.isalpha():
                token_emb[i, 1] = S * 1.0 
                token_emb[i, 2 + (ord(c) - ord('a'))] = S * 1.0 
                
        for net in range(15):
            start = net * dim_per_net
            token_emb[:, start:start+28] = token_emb[:, 0:28]
            
        # Write exactly to the last 28 dimensions of the base 1020
        token_emb[:, 960:988] = token_emb[:, 0:28]
        
        # Semantic mapping in dimensions 1020 to 1046
        for i, token in enumerate(VOCAB):
            for cat_idx, cat in enumerate(SEMANTIC_KEYS):
                if token in SEMANTIC_WORDS[cat]:
                    token_emb[i, 1020 + cat_idx] = S * 5.0
            
        model.token_emb.weight.data.copy_(token_emb)

        pos_emb = torch.zeros(max_seq_len, d_model)
        for p in range(max_seq_len):
            pos_emb[p, 1080] = C
            pos_emb[p, 1081] = -C
            
            pos_emb[p, 28] = S * (p / max_seq_len)
            pos_emb[p, 29] = S * 1.0
            
        for net in range(16):
            start = net * dim_per_net
            pos_emb[:, start+28:start+30] = pos_emb[:, 28:30]
            
        model.pos_emb.weight.data.copy_(pos_emb)

        l1_attn = model.blocks[0].attn
        mlp1 = model.blocks[0].mlp
        
        l2_attn = model.blocks[1].attn
        mlp2 = model.blocks[1].mlp
        
        for net in range(16):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            # --- LAYER 1 ---
            if net < 15:
                l1_decay = 15.0 + float(net) * (65.0 / 14.0) 
            else:
                l1_decay = 120.0
            
            l1_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l1_attn.W_k.weight[h_start + 0, d_start + 28] = l1_decay
            
            if net < 15:
                for i in range(28):
                    l1_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                    l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75
            else:
                for i in range(26):
                    l1_attn.W_v.weight[h_start + i, 1020 + i] = 1.0
                    l1_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.75
                
            std_dev = 0.5
            nn.init.normal_(mlp1.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev)
            nn.init.normal_(mlp1.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev * S)
            
            # --- LAYER 2 ---
            if net < 15:
                l2_decay = 0.01 + float(net) * (13.99 / 14.0)
            else:
                l2_decay = 10.0
            
            l2_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l2_attn.W_k.weight[h_start + 0, d_start + 28] = l2_decay
            
            if net < 15:
                net_b = (net + 6) % 15
                net_c = (net + 12) % 15
                d_start_b = net_b * dim_per_net
                d_start_c = net_c * dim_per_net
                
                for i in range(22):
                    l2_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                    l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.15
                    
                for i in range(21):
                    l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
                    l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                    
                for i in range(21):
                    l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
                    l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 0.85
                    
                std_dev2 = 0.01 + float(net) * (3.99 / 14.0)
            else:
                for i in range(26):
                    l2_attn.W_v.weight[h_start + i, d_start + 30 + i] = 1.0
                    l2_attn.W_o.weight[d_start + 30 + i, h_start + i] = S * 1.5
                std_dev2 = 0.5
            
            nn.init.normal_(mlp2.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev2)
            nn.init.normal_(mlp2.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev2 * S)

        # Zero out MLP and Attention for dimensions 960-988 to keep exact tokens pure
        model.blocks[0].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[0].mlp.fc2.weight.data[960:988, :] = 0
        model.blocks[1].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[1].mlp.fc2.weight.data[960:988, :] = 0

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
                
        nn.init.ones_(model.final_ln.weight)
        nn.init.zeros_(model.final_ln.bias)
        model.final_ln.bias.data += 1.18
        model.final_ln.bias.data += 1.18

model_shorthand_name = "Interpretable_Semantic_Lexicon"
model_description = "Breaks the physical geometric ceiling by incorporating a hardcoded, 26-axis semantic lexicon. Maps 300+ English words to precise conceptual clusters (e.g., motion, emotion, time) and integrates them over long temporal decays."

def build_embedder(device: str = 'cuda',
                   d_model: int = 1088, n_heads: int = 17, n_layers: int = 2,
                   d_ff: int = 4096, max_seq_len: int = 64) -> InterpretableEmbedder:
    model = SimpleTransformer(
        vocab_size=len(VOCAB), max_seq_len=max_seq_len,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    write_weights(model)
    model.eval()
    return InterpretableEmbedder(model, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n--- Testing model: {model_shorthand_name} ---")
    print(model_description)

    embedder = build_embedder(device=args.device)
    
    config = EncodingConfig()
    config.subject = args.subject
    config.num_train = args.num_train
    config.num_test = args.num_test

    t0 = time.time()
    try:
        results = run_encoding(embedder, config)
        test_corr = results["test_corr"]
        print(f"Mean test correlation: {test_corr:.4f}")
        
        n_params = sum(p.numel() for p in embedder.model.parameters())
        row = make_result_row(results, model_shorthand_name, n_params, model_description, "success")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during evaluation: {e}")
        row = {
            "subject": args.subject,
            "test_corr": 0.0, "train_corr": 0.0, "frac_test_voxels_above_0.2": 0.0,
            "encoding_seconds": time.time() - t0,
            "status": "error", "model_shorthand_name": model_shorthand_name,
            "n_params": sum(p.numel() for p in embedder.model.parameters()),
            "description": model_description,
            "corrs_test_frac>0.1": 0.0, "corrs_test_frac>0.05": 0.0, "corrs_test_frac>0.0": 0.0,
            "corrs_test_median": 0.0, "corrs_test_p75": 0.0, "corrs_test_p90": 0.0, "corrs_test_p95": 0.0, "corrs_test_p99": 0.0
        }

    upsert_overall_results([row], RESULTS_DIR)
