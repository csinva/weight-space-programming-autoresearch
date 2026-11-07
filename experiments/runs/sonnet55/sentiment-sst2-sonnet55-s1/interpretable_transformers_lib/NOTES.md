Snapshot status (honest):
- Snapshots in this folder are NOT full runnable copies: LexiconShiftWindow_v1.py and NegationFlip_v50_cleaned.py are
  descriptive stubs (design + constants). v2..v49 were not saved at all.
- The authoritative, runnable source of the final model (NegationFlip_v50_cleaned, acc 0.785, 5.49e5 params) is
  ../interpretable_transformer.py in the run folder.
Best accuracy seen: 0.790 (v32, v36, v37, v41-v43), within noise of v50 on 200 examples.
