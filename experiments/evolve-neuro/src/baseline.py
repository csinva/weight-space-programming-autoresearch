"""Pretrained baseline embedder (GPT-2 XL).

This is the strong reference point that the hand-built interpretable transformer
in `interpretable_transformer.py` tries to match/approach. It is run once by
`setup_run.py` to populate the baseline row of `results/overall_results.csv`.

`GPT2Embedder` maps each input string to the hidden state of its final token at
a chosen layer. It exposes the embedder interface used by the encoding pipeline:
`__call__(texts: list[str]) -> np.ndarray` of shape (n_texts, hidden_dim).
"""
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class GPT2Embedder:
    """Wraps GPT-2 XL to return the final-token hidden state at a given layer."""

    def __init__(self, checkpoint='gpt2-xl', layer=24, device='cuda'):
        self.layer = layer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            checkpoint, output_hidden_states=True, torch_dtype=torch.float16
        ).to(device).eval()

    @torch.no_grad()
    def __call__(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc='embedding'):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors='pt', padding=True, truncation=True, max_length=128
            ).to(self.device)
            # (n_layers+1) tuple of (batch, tokens, hidden)
            hidden = self.model(**inputs).hidden_states[self.layer]
            # gather the last non-pad token for each sequence (robust to padding side)
            last_idx = inputs['attention_mask'].sum(dim=1) - 1
            emb = hidden[torch.arange(hidden.shape[0]), last_idx]
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0)
