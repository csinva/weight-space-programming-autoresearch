from test_wb_recreate import build_wb_exact
from src.eval import run_encoding, EncodingConfig, make_result_row, upsert_overall_results

embedder = build_wb_exact()
cfg = EncodingConfig(subject="UTS03", num_train=8, num_test=2)
r = run_encoding(embedder, cfg)
n_params = sum(p.numel() for p in embedder.model.parameters())

print(f"train_corr={r['corrs_train_mean']:.4f} test_corr={r['test_corr']:.4f}")

upsert_overall_results(
    [make_result_row(r, "ExactWordBoundaryRepro", n_params, "Exact reproduction of 0.0405 model with d_model=64")], 
    "results")
