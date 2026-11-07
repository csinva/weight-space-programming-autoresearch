import torch
import torch.nn as nn
from top_words import TOP_WORDS

_VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789'-.!()[]{}\\"
VOCAB = ['<pad>', '<unk>'] + list(_VOCAB_CHARS)

print(TOP_WORDS[:10])
