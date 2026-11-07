from src.eval import make_result_row, upsert_overall_results
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
r = {
    'subject': 'UTS03',
    'test_corr': 0.1185,
    'corrs_train_mean': 0.0,
    'corrs_test_frac>0.2': 0.0,
    'encoding_seconds': 0.0,
    'roi_corrs': {}
}
row = make_result_row(
    r=r,
    status="success",
    model_shorthand_name="SuperEmbedding_2Models_1Layer",
    n_params=2.2e10,
    description="Llama-3 8B (16), Qwen 14B (24).",
)
upsert_overall_results([row], RESULTS_DIR)
print("Saved manually!")
