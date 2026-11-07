"""Parallel config sweep over interpretable_transformer globals.

Usage: CUDA_VISIBLE_DEVICES=<g> uv run sweep.py <shard_idx> <n_shards> <configs_module>
Writes one row per config to results/sweep_shard_<shard_idx>.csv (no shared-CSV race).
Each config is a dict of global overrides applied before build_embedder().
"""
import os, sys, csv, json, time

SHARD = int(sys.argv[1]); NSHARD = int(sys.argv[2]); CFG_FILE = sys.argv[3]

import numpy as np
import interpretable_transformer as it

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RESULTS, f"sweep_{CFG_FILE}_shard_{SHARD}.csv")

# baseline globals snapshot (the v47 best) to reset between configs
BASE = {k: getattr(it, k) for k in [
    "LSA_DIM", "LSA_WINDOW", "LSA_DIRECTION", "SPPMI_SHIFT", "LSA2_DIM", "LSA2_WINDOW", "LSA2_DIRECTION",
    "LSA3_DIM", "LSA3_WINDOW", "LSA3_DIRECTION",
    "TOPIC_DIM", "CAT_SCALE", "IDENT_TOPK", "IDENT_SCALE", "HASH_DIM",
    "HASH_LO", "HASH_HI", "HASH_SCALE", "MAXPOOL_DIM", "PREV_DIM", "USE_MORPH",
    "ORTHO_DIM", "ORTHO_SCALE", "RECENCY_LAMBDA", "RAW_COOC_N", "RAW_COOC_DIR", "USE_PHONO",
    "WORDNET_MINW", "WORDNET_SCALE", "WORDNET_LEX", "WORDNET_NSENSES",
]}


def recompute_derived():
    it.N_CATEGORIES = len(it.SEMANTIC_CATEGORIES)
    it.MAIN_DIM = it.RAW_COOC_N if getattr(it, "RAW_COOC_N", 0) else it.LSA_DIM
    it.WORDNET_DIM = (it._wordnet_features(it.VOCAB, it.WORDNET_MINW, it.WORDNET_LEX,
                                           it.WORDNET_NSENSES).shape[1]
                      if it.WORDNET_MINW else 0)
    it.SIG_DIM = (it.MAIN_DIM + it.TOPIC_DIM + it.N_CATEGORIES + it.N_SCALAR
                  + (it.N_MORPH if it.USE_MORPH else 0) + (it.N_PHONO if it.USE_PHONO else 0)
                  + it.WORDNET_DIM + it.IDENT_TOPK + it.HASH_DIM + it.ORTHO_DIM
                  + it.LSA2_DIM + it.LSA3_DIM)
    it.INTERACT_SPECS = [(it.MAIN_DIM + it.TOPIC_DIM, it.N_CATEGORIES)]
    it.INTERACT_DIM = sum(m for _, m in it.INTERACT_SPECS)
    it.MAXPOOL_OFFSET = it.MAIN_DIM + it.TOPIC_DIM


def run_one(cfg):
    for k, v in BASE.items():
        setattr(it, k, v)
    for k, v in cfg.items():
        if k != "name":
            setattr(it, k, v)
    recompute_derived()
    t0 = time.time()
    embedder = it.build_embedder(device="cuda", d_model=2 * it.SIG_DIM)
    ntr = cfg.get("NUM_TRAIN", 8)
    # nboots: bootstrap-CV folds for alpha selection (dominant cost). Default 5
    # (official). Configs may set nboots=2 for ~2x-faster TUNING; confirm the
    # winner at nboots=5. chunklen/nchunks left at eval defaults.
    nboots = cfg.get("NBOOTS", 5)
    ecfg = it.EncodingConfig(subject="UTS03", num_train=ntr, num_test=3, nboots=nboots)
    r = it.run_encoding(embedder, ecfg, verbose=False)
    return {
        "name": cfg["name"],
        "test_corr": round(r["test_corr"], 4),
        "train_corr": round(r["corrs_train_mean"], 4),
        "frac": round(r["corrs_test_frac>0.2"], 4),
        "secs": round(time.time() - t0, 1),
        "cfg": json.dumps({k: v for k, v in cfg.items() if k != "name"}),
    }


CONFIGS = __import__(CFG_FILE).CONFIGS
mine = [c for i, c in enumerate(CONFIGS) if i % NSHARD == SHARD]
print(f"shard {SHARD}: {len(mine)} configs", flush=True)
cols = ["name", "test_corr", "train_corr", "frac", "secs", "cfg"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for c in mine:
        try:
            row = run_one(c)
        except Exception as e:
            row = {"name": c["name"], "test_corr": "", "train_corr": "",
                   "frac": "", "secs": "", "cfg": f"ERROR: {e}"}
        w.writerow(row); f.flush()
        print(f"  {row['name']}: {row['test_corr']}", flush=True)
print(f"shard {SHARD} done", flush=True)
