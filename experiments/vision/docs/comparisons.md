# Baselines and natural-language compression

[README](../README.md) · [Operations](operations.md) · [Scientific protocol](protocol.md) · [Comparisons](comparisons.md)

Choose reference settings and compression budgets before inspecting final scores. Held-out subjects, held-out images, TVSD responses and final artifacts must never guide search, repair, coordination or model selection. Run commands from the repository root.

## Baselines

`baselines/` holds fixed reference models, each a `MODELS` dict that follows the
candidate output contract: pixels/low-level statistics, Gabor wavelet pyramid
(Kay et al., 2008) with and without divisive normalization, wavelet scattering
(Mallat), Portilla–Simoncelli texture statistics on a 2×2 grid (simplifying
Freeman & Simoncelli, 2011), banana-filter curvature (a proxy for Pasupathy &
Connor), GIST, HMAX, and ImageNet-trained and seeded-random AlexNet layers.
They run in-process as trusted code, with the same evaluator, and write to
`final_results/baselines/`, never to a search run. Their settings are fixed a
priori or chosen on search data only.

```bash
pip install -e '.[baselines]'
python -m baselines.run --dataset laion --region visual            # search protocol
python -m baselines.run --dataset laion --region visual --final    # held-out LAION subjects and images
python -m baselines.run --dataset nsd --final --model champion=final_results/.../model.py  # any champion on NSD
python -m baselines.run --dataset triplen --only gabor,hmax
```

## Natural-language compression

`compress.py` tests whether a model's broad computation can be specified in
words. Claude (Opus 5.5, medium effort, no tools or memory) describes the model's
code in at most N words. Independent Claude calls rebuild a model from that text
alone, with only a contract check as feedback. The original, the rebuilds
and the `gabor_norm` floor are then scored with the search evaluator. The report
gives score retention, (rebuild − floor) / (original − floor), and linear CKA with
the original's features. Outputs go to `final_results/compression/`.

```bash
python compress.py --model path/to/model.py --words 200 --builders 4
python compress.py --model path/to/model.py --words 50,100,200,400 --final
python compress.py --model path/to/model.py --description "A V1-like Gabor energy model."
```

Compression currently scores LAION visual encoding; `--final` adds the held-out LAION evaluation. Both comparison entry points use `neuro_eval/comparison.py` for shared scoring.
