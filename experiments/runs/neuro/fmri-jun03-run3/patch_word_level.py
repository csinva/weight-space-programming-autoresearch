import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
with open(filepath, "r") as f:
    content = f.read()

# Replace VOCAB definition
vocab_replacement = """
from top_words import TOP_WORDS
VOCAB_WORDS = TOP_WORDS[:2000]
VOCAB = ['<pad>', '<unk>'] + VOCAB_WORDS
"""
content = re.sub(
    r"_VOCAB_CHARS = .*?\nVOCAB = \['<pad>', '<unk>'\] \+ list\(_VOCAB_CHARS\)",
    vocab_replacement,
    content,
    flags=re.DOTALL
)

# Replace encode_text logic
encode_replacement = """
    def encode_text(self, text: str) -> torch.Tensor:
        words = text.lower().split()
        words = words[-self.model.max_seq_len:]
        
        # pad left
        while len(words) < self.model.max_seq_len:
            words.insert(0, '<pad>')
            
        ids = []
        for w in words:
            if w in VOCAB:
                ids.append(VOCAB.index(w))
            else:
                ids.append(VOCAB.index('<unk>'))
        return torch.tensor(ids, dtype=torch.long)

    @torch.no_grad()
    def forward(self, input_strings: List[str]) -> torch.Tensor:
        B = len(input_strings)
        T = self.model.max_seq_len
        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, s in enumerate(input_strings):
            input_ids[i] = self.encode_text(s)
        hidden_states = self.model(input_ids)
        return hidden_states[:, -1, :].cpu()
"""
content = re.sub(
    r"    @torch\.no_grad\(\)\n    def forward\(self, input_strings: List\[str\]\) -> torch\.Tensor:.*?return hidden_states\[:, -1, :\]\.cpu\(\)",
    encode_replacement,
    content,
    flags=re.DOTALL
)

with open(filepath, "w") as f:
    f.write(content)
