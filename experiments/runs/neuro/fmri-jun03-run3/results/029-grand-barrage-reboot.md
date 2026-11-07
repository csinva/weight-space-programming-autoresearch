# Checkpoint 029: Grand Barrage Reboot

## Issue
During the initial launch of the Grand Barrage, we hit two significant roadblocks:
1. `Qwen 14B TripleScale` hit a CUDA OOM because the sequence length (B=16) was too large for an architecture calculating and extracting three distinct sets of dense embeddings.
2. The sheer amount of overlapping Ridge Regression models computing in parallel via SVD starved the CPU of compute threads, causing `eval.py` to hang indefinitely after extracting features and saving models.

## Resolution
We killed the hanging SVD evaluations (`pkill -f eval_`) which terminated the models that were hung inside `np.linalg.svd`. 
We then re-launched the models sequentially or batched properly across the isolated GPUs to avoid thread saturation and OOM errors:
- `Qwen 7B TripleScale`
- `Llama-3 8B MultiScale`
- `Ensemble Qwen1.5B + GPT-2-XL`
- `Mistral-7B MultiScale`

We are now waiting for the ridge regression matrices to evaluate.
