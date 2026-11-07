import torch
import torch.nn as nn
import numpy as np
import os
import sys

from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PureLlamaEmbedder(nn.Module):
    def __init__(self, hf_model_name="meta-llama/Llama-2-7b-hf"): # Note: likely need auth or a smaller model. Let's try GPT-2 first instead to avoid auth issues if we don't have a token.
        super().__init__()
        pass
