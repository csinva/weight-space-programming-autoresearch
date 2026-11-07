"""Legit feature experiments: phonological (audio), higher orthographic, cat-congruence,
scale tuning — all no-corpus-stats, on the full-vocab legit base, mostly at ntr64/93."""
CONFIGS = []
LEGIT = dict(LSA_DIM=0, TOPIC_DIM=0, LSA2_DIM=0, LSA3_DIM=0, IDENT_TOPK=0,
             HASH_DIM=0, RAW_COOC_N=0, USE_MORPH=True, USE_PHONO=False,
             ORTHO_DIM=200, ORTHO_SCALE=1.0, CAT_SCALE=1.5, NUM_TRAIN=64)


def add(name, **kw):
    c = dict(LEGIT); c.update(kw); c["name"] = name
    CONFIGS.append(c)


# phonological features (audio dataset) at ntr64 and ntr93
add("phono_ntr64", USE_PHONO=True)
add("phono_ntr93", USE_PHONO=True, NUM_TRAIN=93)
add("phono_catsc3_ntr93", USE_PHONO=True, CAT_SCALE=3.0, NUM_TRAIN=93)
# higher orthographic over 12k vocab (cleaner memorization)
for od in [600, 1200]:
    add(f"ortho{od}_ntr93", ORTHO_DIM=od, NUM_TRAIN=93)
# orthographic scale tuning at ntr93
for osc in [0.5, 2.0]:
    add(f"orthosc{osc}_ntr93", ORTHO_SCALE=osc, NUM_TRAIN=93)
# category scale at ntr93
for cs in [2.5, 4.0]:
    add(f"catsc{cs}_ntr93", CAT_SCALE=cs, NUM_TRAIN=93)
# everything legit on at ntr93
add("alllegit_ntr93", USE_PHONO=True, ORTHO_DIM=400, CAT_SCALE=2.5, NUM_TRAIN=93)
add("ref_legit_ntr93", NUM_TRAIN=93)
