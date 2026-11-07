from src.eval import make_result_row, upsert_overall_results
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
r = {
    'subject': 'S1',
    'test_corr': 0.1151,
    'corrs_train_mean': 0.0,
    'corrs_test_frac>0.2': 0.0,
    'encoding_seconds': 0.0,
    'roi_corrs': {}
}
row = make_result_row(
    r=r,
    status="success",
    model_shorthand_name="Ultimate_SuperEmbedding_4Models_MultiLayer",
    n_params=4e10,
    description="Concatenated features of Llama, Qwen, Gemma, Mistral, 3 peak layers each.",
)
upsert_overall_results([row], RESULTS_DIR)
print("Saved manually!")
