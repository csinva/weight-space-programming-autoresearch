"""Legit model with EXPANDED hand-coded categories (57, no corpus stats). ortho200 base."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True,
             ORTHO_DIM=200, ORTHO_SCALE=1.0, CAT_SCALE=1.5, NUM_TRAIN=8)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


# expanded-category legit model scaling
for ntr in [8, 32, 64, 93]:
    add(f"x57_ntr{ntr}", NUM_TRAIN=ntr)
# ablations at ntr64: what carries signal?
add("x57_catsonly_ntr64", ORTHO_DIM=0, NUM_TRAIN=64)
add("x57_orthoonly_ntr64", CAT_SCALE=0.0, USE_MORPH=False, NUM_TRAIN=64)
# category scale tuning with richer set
for cs in [1.0, 2.5, 4.0]:
    add(f"x57_catsc{cs}_ntr64", CAT_SCALE=cs, NUM_TRAIN=64)
# bigger ortho with rich cats at high data
add("x57_ortho400_ntr93", ORTHO_DIM=400, NUM_TRAIN=93)
add("x57_catsc3_ntr93", CAT_SCALE=3.0, NUM_TRAIN=93)
