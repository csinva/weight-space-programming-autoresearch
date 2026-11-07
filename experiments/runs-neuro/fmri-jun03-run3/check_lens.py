import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src import data, features

train_stories, _ = data.get_story_names(8, 2)
wordseqs = data.load_wordseqs(train_stories)

lens = []
for story in train_stories:
    ws = wordseqs[story]
    ngrams = features.get_ngrams(list(ws.data), ngram_size=10)
    for n in ngrams:
        lens.append(len(n))

print(f"Max 10-gram len: {max(lens)}")
print(f"Mean 10-gram len: {np.mean(lens)}")
print(f"99th percentile: {np.percentile(lens, 99)}")
