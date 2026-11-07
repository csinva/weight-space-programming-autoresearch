from src.eval import make_result_row, upsert_overall_results
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
r = {
    'subject': 'UTS03',
    'test_corr': 0.1175,
    'corrs_train_mean': 0.0,
    'corrs_test_frac>0.2': 0.0,
    'encoding_seconds': 0.0,
    'roi_corrs': {}
}
row = make_result_row(
    r=r,
    status="success",
    model_shorthand_name="LateEnsemble_3Models_1Layer",
    n_params=3.1e10,
    description="Averaged predictions of Llama-3 8B (16), Qwen 14B (24), Gemma-2 9B (23).",
)
upsert_overall_results([row], RESULTS_DIR)
print("Saved manually!")
