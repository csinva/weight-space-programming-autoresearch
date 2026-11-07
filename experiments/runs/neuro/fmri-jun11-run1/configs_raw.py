"""Raw (non-SVD) co-occurrence semantic features (Huth-style) at high data.
Does preserving full word-specific PPMI detail beat SVD-compressed LSA when data is plentiful?"""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# raw-cooc main view, N contexts, at ntr64 (right direction)
for n in [300, 500, 700, 985]:
    add(f"raw{n}_ntr64", RAW_COOC_N=n, NUM_TRAIN=64)
# direction test at ntr64
add("raw600both_ntr64", RAW_COOC_N=600, RAW_COOC_DIR="both", NUM_TRAIN=64)
add("raw600left_ntr64", RAW_COOC_N=600, RAW_COOC_DIR="left", NUM_TRAIN=64)
# low-data check: does raw help/hurt at ntr8 and ntr32?
add("raw600_ntr8", RAW_COOC_N=600, NUM_TRAIN=8)
add("raw600_ntr32", RAW_COOC_N=600, NUM_TRAIN=32)
# headline: raw at max data
add("raw600_ntr93", RAW_COOC_N=600, NUM_TRAIN=93)
add("raw985_ntr93", RAW_COOC_N=985, NUM_TRAIN=93)
