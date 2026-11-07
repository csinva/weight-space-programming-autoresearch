import re
from collections import Counter
import os

# Let's see what words are most common in the text data.
data_dir = "/home/chansingh/circuit-building-autoresearch/data/neuro/uts03/Train"
words = Counter()
try:
    for f in os.listdir(data_dir):
        if f.endswith('.txt'):
            with open(os.path.join(data_dir, f)) as txt:
                text = txt.read().lower()
                for w in re.findall(r'\b[a-z]+\b', text):
                    words[w] += 1
    print(words.most_common(20))
except:
    pass
