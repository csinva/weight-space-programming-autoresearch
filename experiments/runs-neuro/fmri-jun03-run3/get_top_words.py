import sys
import os
sys.path.append(os.path.abspath(os.path.join("..", "..", "evolve-neuro", "src")))

import joblib
from collections import Counter
import ridge_utils

wordseqs_path = '/home/chansingh/mntv1/deep-fMRI/data/huge_data/wordseqs.joblib'
try:
    wordseqs = joblib.load(wordseqs_path)
    counter = Counter()
    for story, seq in wordseqs.items():
        for w in seq.data:
            counter[w.lower()] += 1
    
    print("Top 50 words:")
    print([w for w, c in counter.most_common(50)])
    
    vocab_chars = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
    top_words = [w for w, c in counter.most_common(4000) if all(c in vocab_chars for c in w)]
    with open('top_words.py', 'w') as f:
        f.write(f"TOP_WORDS = {top_words}\n")
except Exception as e:
    import traceback
    traceback.print_exc()

