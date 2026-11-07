"""Push orthographic capacity at MAX data (ntr93): cleaner word-identity memorization.
ortho200->0.0735, ortho600->0.0764. No corpus stats."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_SCALE=1.0, CAT_SCALE=1.5, NUM_TRAIN=93)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


for od in [800, 1000, 1500, 2000, 3000]:
    add(f"ortho{od}_ntr93", ORTHO_DIM=od)
# ortho-scale at the larger dim
add("ortho1500_sc0.5", ORTHO_DIM=1500, ORTHO_SCALE=0.5)
add("ortho1500_sc2", ORTHO_DIM=1500, ORTHO_SCALE=2.0)
# bigger ortho + richer cats
add("ortho1500_catsc3", ORTHO_DIM=1500, CAT_SCALE=3.0)
