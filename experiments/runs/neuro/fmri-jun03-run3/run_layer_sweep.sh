#!/bin/bash

# Sweep layers 0, 7, 14, 21, 28 for both Trained and Random, and both Last and Mean pooling
# Total runs: 5 layers * 2 states * 2 poolings = 20 runs.

LAYERS=(0 7 14 21 28)

for L in "${LAYERS[@]}"; do
    # Trained
    uv run runs-neuro/fmri-jun03-run3/eval_llm_vs_random_layer_sweep.py --layer $L
    uv run runs-neuro/fmri-jun03-run3/eval_llm_vs_random_layer_sweep.py --layer $L --mean
    
    # Random
    uv run runs-neuro/fmri-jun03-run3/eval_llm_vs_random_layer_sweep.py --layer $L --random
    uv run runs-neuro/fmri-jun03-run3/eval_llm_vs_random_layer_sweep.py --layer $L --random --mean
done

