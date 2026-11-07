"""Evaluate the cross-run competitor (jun03-run4) embedder IN THIS harness, alone
and stacked with MY within-story novelty block. If their stronger bag (0.0818)
stacks with novelty (+0.0016 orthogonal) it could beat GPT-2 XL (0.0826).
The user explicitly authorized using other runs-neuro folders for ideas.
"""
import os, sys, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.eval import EncodingConfig, run_encoding
import interpretable_transformer as MINE  # for _novelty_block + _ngram_word_lists

COMP_PATH = os.path.join(HERE, '..', 'fmri-jun03-run4', 'interpretable_transformer.py')


def load_competitor():
    spec = importlib.util.spec_from_file_location('comp_it', COMP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['comp_it'] = mod
    spec.loader.exec_module(mod)
    return mod


def my_novelty(ngrams):
    wl = MINE._ngram_word_lists(list(ngrams))
    return MINE._novelty_block(wl)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'comp_only'
    comp = load_competitor()
    base = comp.build_embedder(device='cuda')
    if mode == 'comp_only':
        emb = base
    else:  # comp + novelty
        def emb(ngrams):
            ngrams = list(ngrams)
            bag = np.asarray(base(ngrams), dtype=np.float32)
            nov = my_novelty(ngrams)
            return np.hstack([bag, nov])
    cfg = EncodingConfig(subject='UTS03', num_train=8, num_test=3)
    r = run_encoding(emb, cfg, verbose=False)
    print(f'MODE={mode}  test_corr={r["test_corr"]:.5f}  train_corr={r["corrs_train_mean"]:.5f}')
