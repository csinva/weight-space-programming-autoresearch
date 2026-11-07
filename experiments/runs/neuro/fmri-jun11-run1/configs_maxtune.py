"""Re-tune the stack at higher data (num_train=64): what does MORE DATA unlock?
At 8 stories identK65/topic50/dim160 were optimal; more data may support richer features."""
CONFIGS = []
NT = 64


def add(name, **kw):
    kw.setdefault("NUM_TRAIN", NT)
    CONFIGS.append(dict(name=name, **kw))


add("base64")                                  # current best config at ntr64 (ref)
# more identity words (more samples per word with more data)
for k in [120, 200, 350, 500]:
    add(f"identK{k}", IDENT_TOPK=k)
# more topic dims
for t in [100, 150, 200]:
    add(f"topic{t}", TOPIC_DIM=t)
# bigger 2nd (symmetric) view
for d in [120, 160]:
    add(f"v2d{d}", LSA2_DIM=d, LSA2_WINDOW=5)
# orthographic now (more data may let surface features help)
add("ortho80", ORTHO_DIM=80, ORTHO_SCALE=1.0)
# richer combo: more identity + more topic + bigger 2nd
add("rich1", IDENT_TOPK=200, TOPIC_DIM=150, LSA2_DIM=120)
add("rich2", IDENT_TOPK=350, TOPIC_DIM=200, LSA2_DIM=160, LSA_DIM=200)
