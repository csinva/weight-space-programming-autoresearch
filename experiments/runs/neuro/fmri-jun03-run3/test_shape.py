import numpy as np

def _lanczosfun(cutoff, t, window=3):
    t = t * cutoff
    val = window * np.sin(np.pi * t) * np.sin(np.pi * t / window) / (np.pi ** 2 * t ** 2)
    val[t == 0] = 1.0
    val[np.abs(t) > window] = 0.0
    return val

def lanczos_downsample(data, oldtime, newtime, window=3):
    cutoff = 1 / np.mean(np.diff(newtime))
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for n in range(len(newtime)):
        sincmat[n, :] = _lanczosfun(cutoff, newtime[n] - oldtime, window)
    return np.dot(sincmat, data)

from src import data
from src.eval import EncodingConfig

cfg = EncodingConfig()
train_stories, test_stories = data.get_story_names(cfg.num_train, cfg.num_test)
wordseqs = data.load_wordseqs(train_stories + test_stories)
responses = data.load_responses(train_stories + test_stories, subject=cfg.subject)

y_shapes = []
x_shapes = []
for s in train_stories:
    resp = responses[s]
    y_shapes.append(resp.shape[0])
    
    n_words = len(wordseqs[s].data)
    X = np.zeros((n_words, 150))
    ds = lanczos_downsample(X, wordseqs[s].data_times, wordseqs[s].tr_times)
    x_shapes.append(ds.shape[0])

print(f"Y total: {sum(y_shapes)}")
print(f"X total: {sum(x_shapes)}")

from src.features import make_delayed
ds_delayed = make_delayed(ds, 4)
print(f"Delayed X: {ds_delayed.shape}")
