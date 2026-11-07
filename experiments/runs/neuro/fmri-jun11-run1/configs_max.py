"""Max-data headline runs: best model at num_train=93 (all stories), dim 160 vs 300."""
CONFIGS = [
    dict(name="mymax_ntr93_dim160", NUM_TRAIN=93, LSA_DIM=160),
    dict(name="mymax_ntr93_dim300", NUM_TRAIN=93, LSA_DIM=300),
]
