import os
import torch
from interpretable_transformer import build_embedder, model_shorthand_name, model_description
from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n--- Testing model: {model_shorthand_name} ---")
print(model_description)

embedder = build_embedder(device=device)

config = EncodingConfig(
    subject="UTS03",
    num_train=8,
    num_test=2,
    ngram_size=10,
    ndelays=4,
    nboots=5,
    chunklen=40,
    nchunks=20,
    trim_edges=True
)

results = run_encoding(embedder, config, verbose=True)

n_params = sum(p.numel() for p in embedder.model.parameters())
row = make_result_row(results, model_shorthand_name, n_params, model_description)
upsert_overall_results([row], RESULTS_DIR)

print(f"Mean test correlation: {results['test_corr']:.4f}")
