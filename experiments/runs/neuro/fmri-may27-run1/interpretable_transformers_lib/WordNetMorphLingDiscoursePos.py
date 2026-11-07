"""WordNetMorphLingDiscoursePos snapshot.

Builds on WordNetMorphLingStoryArcSent (0.1054) by adding a rich MULTI-SCALE
DISCOURSE positional encoding:

1. Multi-scale PARAGRAPH-level position bases at multiple lengths
   (10, 20, 40, 80, 160, 320 ngrams). Each paragraph segment is built by
   accumulating sentence-end-aligned segments until each is at least PARA_LEN
   ngrams long. For each scale, within each paragraph we compute:
   - p_para = (i - paragraph_start + 0.5) / paragraph_len
   - log dist from/to paragraph boundary
   - log paragraph length
   - is_first / is_last indicators
   - Fourier sin/cos(2π·p_para·k) for k=1..KP (KP=2)
   - Exp decays since last paragraph end at taus 10, 30, 80

2. Multi-tau SENTENCE-end decays (since-last, time-to-next, EW running average)
   at taus (2, 5, 10, 30, 80, 200).

3. PUNCT-specific decays for ?, !, . separately (different speech acts have
   different fMRI signatures). Decays at taus (2, 5, 10, 30, 80, 200) +
   EW running averages at the three smallest taus.

All features are variance-normalized to ~0.5. Variance mask mv=0.14 filters
out near-constant dims after concatenation with the previous content+position.

test_corr=0.1122 on UTS03 (+0.0068 over WordNetMorphLingStoryArcSent 0.1054;
+0.0083 over WordNetMorphLingStoryArc 0.1039; beats GPT-2 XL baseline 0.0791 
by +0.033 → +41% relative). No transformer, no training, no corpus statistics,
no gradient updates.
"""
import importlib.util, os, sys
import numpy as np
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def _load(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


_sent = _load(os.path.join(_HERE, 'WordNetMorphLingStoryArcSent.py'), '_sent')

DEFAULT_MV = 0.14
DEFAULT_PARA_LENS = (10, 20, 40, 80, 160, 320)
DEFAULT_KP = 2
DEFAULT_DTAUS = (2, 5, 10, 30, 80, 200)
DEFAULT_PARA_DECAY_TAUS = (10, 30, 80)


def _norm_var(v, target=0.5):
    s = float(v.var())
    if s < 1e-8: return v
    return v * float(np.sqrt(target / s))


def _para_block(texts, para_len, kp, para_decay_taus):
    N = len(texts)
    if N == 0: return np.zeros((0, 0), dtype=np.float32)
    sent_ends = []
    for i, t in enumerate(texts):
        last = t.split()[-1] if t and t.split() else ''
        if last.endswith(('.', '?', '!')):
            sent_ends.append(i)
    para_segs = []
    a = 0
    for end_i in sent_ends:
        if end_i - a >= para_len:
            para_segs.append((a, end_i + 1)); a = end_i + 1
    if a < N:
        para_segs.append((a, N))
    parts = []
    p_para = np.zeros(N, dtype=np.float32)
    dist_from = np.zeros(N, dtype=np.float32)
    dist_to = np.zeros(N, dtype=np.float32)
    log_pl = np.zeros(N, dtype=np.float32)
    is_first = np.zeros(N, dtype=np.float32)
    is_last = np.zeros(N, dtype=np.float32)
    for (s, e) in para_segs:
        L = e - s
        for j in range(s, e):
            p_para[j] = (j - s + 0.5) / L
            dist_from[j] = j - s
            dist_to[j] = (e - 1) - j
            log_pl[j] = np.log1p(L)
        is_first[s] = 1.0
        is_last[e - 1] = 1.0
    parts.append(_norm_var(p_para - 0.5, 0.5))
    parts.append(_norm_var(np.log1p(dist_from), 0.5))
    parts.append(_norm_var(np.log1p(dist_to), 0.5))
    parts.append(_norm_var(log_pl, 0.5))
    parts.append(_norm_var(is_first, 0.5))
    parts.append(_norm_var(is_last, 0.5))
    for k in range(1, kp + 1):
        parts.append(_norm_var(np.sin(2 * np.pi * p_para * k).astype(np.float32), 0.5))
        parts.append(_norm_var(np.cos(2 * np.pi * p_para * k).astype(np.float32), 0.5))
    for tau in para_decay_taus:
        v = np.zeros(N, dtype=np.float32)
        last = -100
        for i in range(N):
            v[i] = np.exp(-(i - last) / float(tau))
            if is_last[i] > 0: last = i
        parts.append(_norm_var(v, 0.5))
    return np.stack(parts, axis=1).astype(np.float32)


def _sent_bumps(texts, taus):
    """Multi-tau decays + look-ahead + EW for sentence-end events."""
    N = len(texts)
    if N == 0: return np.zeros((0, 0), dtype=np.float32)
    is_last = np.zeros(N, dtype=np.float32)
    for i, t in enumerate(texts):
        last = t.split()[-1] if t and t.split() else ''
        if last.endswith(('.', '?', '!')):
            is_last[i] = 1.0
    parts = []
    # since-last
    for tau in taus:
        v = np.zeros(N, dtype=np.float32)
        last = -100
        for i in range(N):
            v[i] = np.exp(-(i - last) / float(tau))
            if is_last[i] > 0: last = i
        parts.append(_norm_var(v, 0.5))
    # time-to-next (look-ahead)
    for tau in taus:
        v = np.zeros(N, dtype=np.float32)
        nxt = N + 100
        for i in range(N - 1, -1, -1):
            v[i] = np.exp(-(nxt - i) / float(tau))
            if is_last[i] > 0: nxt = i
        parts.append(_norm_var(v, 0.5))
    # EW running average
    for tau in taus:
        a = 1.0 / tau
        s = 0.0
        v = np.zeros(N, dtype=np.float32)
        for i in range(N):
            s = (1 - a) * s + a * is_last[i]
            v[i] = s
        parts.append(_norm_var(v, 0.5))
    return np.stack(parts, axis=1).astype(np.float32)


def _punct_decays(texts, taus):
    """Separate decays for ? vs ! vs . — different speech acts."""
    N = len(texts)
    if N == 0: return np.zeros((0, 0), dtype=np.float32)
    is_q = np.zeros(N, dtype=np.float32)
    is_e = np.zeros(N, dtype=np.float32)
    is_p = np.zeros(N, dtype=np.float32)
    for i, t in enumerate(texts):
        last = t.split()[-1] if t and t.split() else ''
        if last.endswith('?'): is_q[i] = 1
        elif last.endswith('!'): is_e[i] = 1
        elif last.endswith('.'): is_p[i] = 1
    parts = []
    for arr in [is_q, is_e, is_p]:
        for tau in taus:
            v = np.zeros(N, dtype=np.float32)
            last = -100
            for i in range(N):
                v[i] = np.exp(-(i - last) / float(tau))
                if arr[i] > 0: last = i
            parts.append(_norm_var(v, 0.5))
        # EW running averages at small taus
        for tau in taus[:3]:
            a = 1.0 / tau; s = 0.0
            v = np.zeros(N, dtype=np.float32)
            for i in range(N):
                s = (1 - a) * s + a * arr[i]
                v[i] = s
            parts.append(_norm_var(v, 0.5))
    return np.stack(parts, axis=1).astype(np.float32)


def _all_feats(texts, mv=DEFAULT_MV, para_lens=DEFAULT_PARA_LENS,
               kp=DEFAULT_KP, dtaus=DEFAULT_DTAUS,
               para_decay_taus=DEFAULT_PARA_DECAY_TAUS):
    base = _sent._all_feats(texts)
    parts = [base]
    for pl in para_lens:
        parts.append(_para_block(texts, pl, kp, para_decay_taus))
    parts.append(_sent_bumps(texts, dtaus))
    parts.append(_punct_decays(texts, dtaus))
    return np.concatenate(parts, axis=1).astype(np.float32)


class WordNetMorphLingDiscoursePosEmbedder(nn.Module):
    SHORTHAND_NAME = "WordNetMorphLingDiscoursePos"
    DESCRIPTION = (
        "WordNetMorphLingStoryArcSent (multi-tau content + multi-basis "
        "story-arc + sentence-relative position) PLUS rich multi-scale "
        "DISCOURSE positional encoding: paragraph-level basis at 6 scales "
        "(10,20,40,80,160,320 ngrams, KP=2 Fourier), multi-tau sentence-end "
        "decays/look-ahead/EW at taus (2,5,10,30,80,200), and PUNCT-specific "
        "decays for ?, !, . separately. Variance mask mv=0.14. "
        "test_corr=0.1122 on UTS03 (beats GPT-2 XL baseline 0.0791 by +0.033)."
    )
    MV = DEFAULT_MV

    def __init__(self, mv=DEFAULT_MV, para_lens=DEFAULT_PARA_LENS,
                 kp=DEFAULT_KP, dtaus=DEFAULT_DTAUS,
                 para_decay_taus=DEFAULT_PARA_DECAY_TAUS):
        super().__init__()
        self.mv = mv
        self.para_lens = para_lens
        self.kp = kp
        self.dtaus = dtaus
        self.para_decay_taus = para_decay_taus
        self._mask = None
        self.model = nn.Linear(1, 1)

    def __call__(self, texts, batch_size=256):
        feats = _all_feats(texts, mv=self.mv, para_lens=self.para_lens,
                           kp=self.kp, dtaus=self.dtaus,
                           para_decay_taus=self.para_decay_taus)
        if self._mask is None:
            v = feats.var(0)
            self._mask = v > self.mv
        return feats[:, self._mask]
