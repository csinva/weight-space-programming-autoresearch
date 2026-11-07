import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time

sys.path.insert(0, os.path.dirname(__file__))

from src import data, features, encoding
from src.eval import run_encoding, make_result_row, upsert_overall_results, EncodingConfig
from interpretable_transformer import SimpleTransformer, InterpretableEmbedder, VOCAB

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SimpleTransformer(
    vocab_size=len(VOCAB), max_seq_len=64,
    d_model=1020, n_heads=10, n_layers=2, d_ff=4000)
model.load_state_dict(torch.load("runs-neuro/fmri-jun03-run3/trained_transformer.pt"))
model.eval()

embedder = InterpretableEmbedder(model, device=device)

cfg = EncodingConfig()
cfg.subject = "UTS03"
cfg.num_train = 8
cfg.num_test = 2

t0 = time.time()
results = run_encoding(embedder, cfg)
test_corr = results["test_corr"]
print(f"Mean test correlation: {test_corr:.4f}")

n_params = sum(p.numel() for p in embedder.model.parameters())
row = make_result_row(results, "End_to_End_Trained_10Epochs", n_params, "First attempt at backpropping through the entire ridge regression pipeline.", "success")
upsert_overall_results([row], RESULTS_DIR)
