"""Data-scaling test: does test_corr jump with more training stories (num_train)?
And do higher LSA dims (which overfit at 8 stories) win once there is more data?"""
CONFIGS = []


def add(name, **kw):
    CONFIGS.append(dict(name=name, **kw))


# best config (right-main d160) across training-set sizes
for ntr in [8, 16, 24, 32, 48, 64]:
    add(f"best_ntr{ntr}", NUM_TRAIN=ntr)

# high-dim LSA enabled by more data: at ntr=32 and 48, sweep main dim up
for ntr in [32, 48]:
    for d in [250, 400, 600]:
        add(f"lsadim{d}_ntr{ntr}", NUM_TRAIN=ntr, LSA_DIM=d)

# more identity words also affordable with more data
for ntr in [32, 48]:
    add(f"ident200_ntr{ntr}", NUM_TRAIN=ntr, IDENT_TOPK=200)
