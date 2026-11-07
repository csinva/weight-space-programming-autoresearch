# Fine Layer Sweeps

We are running a few more targeted sweeps to verify the peak representations.

Currently evaluating:
1. Qwen 1.5B layer sweep (Layers 11-17) using context 20.
2. Qwen 1.5B layer sweep (Layers 12-15) using context 20 and the full 8 train/2 test split to perfectly match the SOTA config.
3. Mistral 7B layer sweep (Layers 8, 16, 24, 32) using context 20.

Will update this file when the async scripts finish.

## Results: Qwen 1.5B Fine Layer Sweep (Context 20, 5/1 split)
- **Layer 11 Last**: 0.0850
- **Layer 12 Last**: 0.0786
- **Layer 13 Last**: 0.0816
- **Layer 14 Last**: 0.0888
- **Layer 15 Last**: 0.0880
- **Layer 16 Last**: 0.0906
- **Layer 17 Last**: 0.0876

*Note: In the 5/1 split sweep, Layer 16 actually outperformed Layer 14 slightly.*


## Results: Qwen 1.5B Fine Layer Sweep (Context 20, 8/2 full split)
- **Layer 12 Last**: 0.0917
- **Layer 13 Last**: 0.0968
- **Layer 14 Last**: 0.1028 (!!!)

Wait! Under the full 8/2 train/test split, Layer 14 Last reaches an incredible `0.1028` on its own, *without* ensembling with Mistral!

We need to check the exact `0.0988` test logic. In `0.0988`, we used `Qwen L14 Last` + `Mistral L16Last+L32Mean`. 
If `Qwen L14 Last` achieves `0.1028` on its own, then adding Mistral actually *degraded* the performance from `0.1028` to `0.0988`!

## Results: Qwen 1.5B Fine Layer Sweep (Context 20, 8/2 full split)
- **Layer 12 Last**: 0.0917
- **Layer 13 Last**: 0.0968
- **Layer 14 Last**: 0.1028
- **Layer 15 Last**: 0.1022

Layer 14 Last alone achieves `0.1028`! This breaks the `0.1000` barrier for a single model!
