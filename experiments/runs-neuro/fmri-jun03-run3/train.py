import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src import data, features, encoding
from src.eval import EncodingConfig
from interpretable_transformer import SimpleTransformer, VOCAB

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = EncodingConfig()
cfg.num_train = 8
cfg.num_test = 2
train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
wordseqs = data.load_wordseqs(train_stories)
resps = data.load_responses(train_stories, subject=cfg.subject)

extra_trim = cfg.edge_trim_trs if cfg.trim_edges else 0

if extra_trim:
    resps = {s: r[extra_trim:-extra_trim] for s, r in resps.items()}
resp_train_np = np.vstack([resps[s] for s in train_stories])
resp_train_t = torch.tensor(resp_train_np, dtype=torch.float32, device=device) # (Total_TRs, nvoxels)

class DifferentiableEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        import subprocess
        os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py runs-neuro/fmri-jun03-run3/interpretable_transformer.py")
        import importlib
        import interpretable_transformer
        importlib.reload(interpretable_transformer)
        
        embedder = interpretable_transformer.build_embedder(device=device, d_model=1020, n_heads=10, d_ff=4000)
        self.model = embedder.model
        self.model.train()
        
    def forward(self, input_strings):
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=device)
        for i, s in enumerate(input_strings):
            for j, char in enumerate(s[-T:]):
                if char in VOCAB:
                    input_ids[i, j] = VOCAB.index(char)
                else:
                    input_ids[i, j] = VOCAB.index('<unk>')
        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :] 

diff_embedder = DifferentiableEmbedder().to(device)

# To normalize correctly, we need the global mean and std of the features!
# We can compute it in a no_grad pass first.
global_feat_mean = None
global_feat_std = None

def get_story_downsampled(embedder, story):
    ws = wordseqs[story]
    ngrams = features.get_ngrams(list(ws.data), ngram_size=cfg.ngram_size)
    
    # Checkpoint or batch smaller
    batch_size = 500
    word_vectors_list = []
    for i in range(0, len(ngrams), batch_size):
        batch = ngrams[i:i+batch_size]
        out = embedder(batch)
        word_vectors_list.append(out)
    word_vectors = torch.cat(word_vectors_list, dim=0) 
    
    newtime = ws.tr_times
    oldtime = ws.data_times
    window = 3
    cutoff = 1 / np.mean(np.diff(newtime))
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for i in range(len(newtime)):
        sincmat[i, :] = features._lanczosfun(cutoff, newtime[i] - oldtime, window)
    
    sincmat_t = torch.tensor(sincmat, dtype=torch.float32, device=device)
    downsampled = torch.matmul(sincmat_t, word_vectors)
    
    trim = 5
    lo = 5 + trim + extra_trim
    hi = trim + extra_trim
    return downsampled[lo:-hi]

print("Precomputing global mean/std and initial ridge weights...")
with torch.no_grad():
    diff_embedder.eval()
    all_down = []
    for story in train_stories:
        d = get_story_downsampled(diff_embedder, story)
        all_down.append(d)
    all_down_t = torch.cat(all_down, dim=0)
    
    global_feat_mean = all_down_t.mean(dim=0, keepdim=True)
    global_feat_std = all_down_t.std(dim=0, keepdim=True)
    global_feat_std[global_feat_std == 0] = 1.0
    
    # Compute z-scored and delayed feats for ridge initialization
    feats_z = (all_down_t - global_feat_mean) / global_feat_std
    n, d = feats_z.shape
    ndelays = cfg.ndelays
    out = []
    for delay in range(1, ndelays + 1):
        dstim = torch.zeros((n, d), dtype=torch.float32, device=device)
        dstim[delay:, :] = feats_z[:-delay, :]
        out.append(dstim)
    initial_feats = torch.cat(out, dim=1).cpu().numpy()

import encoding
wt = encoding._ridge_weights(initial_feats, resp_train_np, alpha=100.0)
readout = nn.Linear(initial_feats.shape[1], resp_train_np.shape[1], bias=False).to(device)
readout.weight.data = torch.tensor(wt.T, dtype=torch.float32, device=device)

diff_embedder.train()
optimizer = torch.optim.AdamW(list(diff_embedder.parameters()) + list(readout.parameters()), lr=1e-4, weight_decay=1e-2)

total_trs = resp_train_t.shape[0]

print("Starting training...")
for epoch in range(10):
    optimizer.zero_grad()
    
    total_loss = 0
    tr_offset = 0
    
    for story in train_stories:
        # Recompute differentiable
        downsampled = get_story_downsampled(diff_embedder, story)
        
        # Z-score using global stats
        feats_z = (downsampled - global_feat_mean) / global_feat_std
        
        # Delay
        n, d = feats_z.shape
        out = []
        for delay in range(1, cfg.ndelays + 1):
            dstim = torch.zeros((n, d), dtype=torch.float32, device=device)
            dstim[delay:, :] = feats_z[:-delay, :]
            out.append(dstim)
        delayed_feats = torch.cat(out, dim=1)
        
        # Predict
        preds = readout(delayed_feats)
        
        # True resps for this story
        story_resps = resps[story]
        story_resps_t = torch.tensor(story_resps, dtype=torch.float32, device=device)
        
        # Loss
        loss = nn.MSELoss(reduction='sum')(preds, story_resps_t) / total_trs # scale correctly
        loss.backward()
        
        total_loss += loss.item()
        
        # Free memory!
        del downsampled, feats_z, delayed_feats, preds, story_resps_t, loss
        torch.cuda.empty_cache()

    nn.utils.clip_grad_norm_(diff_embedder.parameters(), 1.0)
    optimizer.step()
    
    print(f"Epoch {epoch+1}/10, Loss: {total_loss:.6f}")

print("Saving weights...")
torch.save(diff_embedder.model.state_dict(), "runs-neuro/fmri-jun03-run3/trained_transformer.pt")
