import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.features import get_features
from src.data import load_story_wordseqs

wordseqs = load_story_wordseqs(["itsabox"])
print(wordseqs["itsabox"].data[:10])

strings = []
for i in range(10):
    # This is roughly what the embedder sees
    words = wordseqs["itsabox"].data[:i+1]
    strings.append(" ".join(words))
print(strings)
