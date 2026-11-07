"""WordNetMorphLingMultiTau snapshot.

Single-file embedder. Adds MULTI-TIMESCALE EXPONENTIALLY-WEIGHTED RUNNING
AVERAGES over the per-step new-word features. For each step (each new word
added to the sliding 10-gram window), we update EW averages at taus
{5, 20, 100, 500} of:

  - WordNet supersense membership of the new word (N_LEX dims)
  - Morpho-syntactic per-word feats (first 8 dims of the 14-dim per-word vector)
  - Phonological features (D_PHONO dims)
  - Morphological per-word features (full _PER_WORD_DIM dims)
  - 9 psycholinguistic indicators (mental verb, deixis, pronoun, q-word,
    discourse marker, sentence-end)
  - 72 semantic category memberships
  - sentence-end / novelty rates

Plus per-ngram features identical to the WordNetMorphLingPlusMasked snapshot
(WNML-tau50 base + phono mean + tense + mental + deixis + speech-act + disc).

Variance mask is computed on the first train story at MV=2e-3, keeping
~360 of ~1800 dims. test_corr=0.0646 on UTS03.
"""
import importlib.util, os, sys
import numpy as np
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

def _load(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

_wnml = _load(os.path.join(_HERE, 'WordNetMorphLing-tau50.py'), '_wnml')
_wnf  = _load(os.path.join(_HERE, 'WordNetFeats-noTransformer.py'), '_wnf')

import interpretable_transformer as IT
_WORD2CAT = IT._WORD2CAT
_CAT2DIM  = IT._CAT2DIM
N_SEM     = IT.N_SEM_CATS
N_LEX     = _wnf.N_LEX

# ---------- phonological (copied from exp11) ----------
from functools import lru_cache

D_PHONO = 10
_VOWELS = set('aeiouy')
_FRICATIVES = set('fvszthsh')
_PLOSIVES = set('bdgkpqt')
_NASALS = set('mn')
_LIQUIDS = set('lr')

@lru_cache(maxsize=20000)
def _phono_feats(w):
    if not w: return None
    w = w.strip(".,;:!?\"'()-").lower()
    if not w: return None
    n_v = sum(1 for c in w if c in _VOWELS)
    n_c = sum(1 for c in w if c.isalpha() and c not in _VOWELS)
    v = np.zeros(D_PHONO, dtype=np.float32)
    L = len(w)
    v[0] = L
    v[1] = n_v
    v[2] = n_c
    v[3] = max(n_v, 1)  # syllable proxy
    v[4] = sum(1 for c in w if c in _FRICATIVES) / max(L, 1)
    v[5] = sum(1 for c in w if c in _PLOSIVES) / max(L, 1)
    v[6] = sum(1 for c in w if c in _NASALS) / max(L, 1)
    v[7] = sum(1 for c in w if c in _LIQUIDS) / max(L, 1)
    v[8] = 1.0 if (L >= 2 and w[-2] == w[-1] and w[-1].isalpha()) else 0.0
    v[9] = 1.0 if (L >= 2 and w[-1] in _VOWELS) else 0.0
    return v

# ---------- tense + mental/deixis/sa/disc (from exp19 + exp21) ----------
_AUX_BE = {'is','was','are','were','am','be','been','being'}
_AUX_HAVE = {'have','has','had','having'}
_MODALS = {'can','could','will','would','shall','should','may','might','must','ought'}

def _tense(words):
    v = np.zeros(5, dtype=np.float32)
    L = max(len(words), 1)
    n_past=n_pres=n_fut=n_perf=n_prog = 0
    for i, w in enumerate(words):
        wl = w.strip(".,;:!?\"'()-").lower()
        if wl in _MODALS: n_fut += 1
        if wl in _AUX_BE:
            n_pres += 1
            if i+1 < len(words):
                nxt = words[i+1].strip(".,;:!?\"'()-").lower()
                if nxt.endswith('ing'): n_prog += 1
        if wl in _AUX_HAVE: n_perf += 1
        if wl.endswith('ed') and len(wl) > 3: n_past += 1
        if wl.endswith('s') and len(wl) > 2 and not wl.endswith('ss'): n_pres += 1
    v[0] = n_past / L
    v[1] = n_pres / L
    v[2] = n_fut  / L
    v[3] = n_perf / L
    v[4] = n_prog / L
    return v


_MENTAL_VERBS = {
    'think','thinks','thought','thinking','know','knows','knew','knowing','known',
    'believe','believes','believed','feel','feels','felt','feeling',
    'guess','guesses','guessed','mean','means','meant','meaning',
    'see','sees','saw','seeing','seen','remember','remembers','remembered','remembering',
    'wonder','wonders','wondered','wondering','understand','understands','understood',
    'imagine','imagines','imagined','realize','realizes','realized',
    'hope','hopes','hoped','want','wants','wanted','wanting','wish','wishes','wished',
    'forget','forgets','forgot','forgotten','suppose','supposes','supposed',
    'expect','expects','expected','assume','assumes','assumed',
    'consider','considers','considered','recognize','recognizes','recognized',
    'doubt','doubts','doubted','seem','seems','seemed',
    'say','says','said','saying','tell','tells','told','telling','ask','asks','asked','asking',
    'reply','replies','replied','whisper','whispers','whispered',
    'mention','mentions','mentioned','explain','explains','explained',
}
_SPATIAL_DEIXIS = {'here','there','this','that','these','those'}
_TEMPORAL_DEIXIS = {'now','then','today','yesterday','tomorrow','soon','later','earlier','recently','currently'}
_PERSON_1 = {'i',"i'm","i've","i'd","i'll",'me','my','mine','myself'}
_PERSON_2 = {'you',"you're","you've","you'd","you'll",'your','yours','yourself','yourselves'}
_PERSON_3 = {'he','she','it','they','him','her','them','his','hers','its','their','theirs','himself','herself','itself','themselves'}
_PERSON_1PL = {'we',"we're","we've","we'd","we'll",'us','our','ours','ourselves'}
_QUESTION_W = {'what','when','where','why','how','who','whom','whose','which'}
_IMPER_HINTS = {"don't",'do','please','just','try','let'}
_DISCOURSE = {'and','but','so','then','because','however','although','though',
              'meanwhile','suddenly','finally','first','next','after','before',
              'while','when','since','until','unless','if'}


def _mental_block(words):
    v = np.zeros(5, dtype=np.float32)
    L = max(len(words), 1)
    for w in words:
        wl = w.strip(".,;:!?\"'()-").lower()
        if wl in _MENTAL_VERBS: v[0] += 1
    return v / L


def _deixis_block(words):
    v = np.zeros(6, dtype=np.float32)
    L = max(len(words), 1)
    for w in words:
        wl = w.strip(".,;:!?\"'()-").lower()
        if wl in _SPATIAL_DEIXIS:  v[0] += 1
        if wl in _TEMPORAL_DEIXIS: v[1] += 1
        if wl in _PERSON_1:        v[2] += 1
        if wl in _PERSON_2:        v[3] += 1
        if wl in _PERSON_3:        v[4] += 1
        if wl in _PERSON_1PL:      v[5] += 1
    return v / L


def _speech_act_block(t, words):
    v = np.zeros(4, dtype=np.float32)
    L = max(len(words), 1)
    s = (t or '').lower()
    for w in words:
        wl = w.strip(".,;:!?\"'()-").lower()
        if wl in _QUESTION_W: v[0] += 1
        if wl in _IMPER_HINTS: v[1] += 1
    v[0] /= L
    v[1] /= L
    if '?' in s: v[2] = 1.0
    if '!' in s: v[3] = 1.0
    return v


def _discourse_block(words):
    v = np.zeros(1, dtype=np.float32)
    L = max(len(words), 1)
    for w in words:
        wl = w.strip(".,;:!?\"'()-").lower()
        if wl in _DISCOURSE: v[0] += 1
    return v / L


# ---------- per-ngram block (the 0.0610 base) ----------
def _per_ngram_block(texts):
    N = len(texts)
    base = _wnml._feats(texts, (50.0,))
    phono = np.zeros((N, D_PHONO), dtype=np.float32)
    tense = np.zeros((N, 5), dtype=np.float32)
    mental = np.zeros((N, 5), dtype=np.float32)
    deixis = np.zeros((N, 6), dtype=np.float32)
    sa     = np.zeros((N, 4), dtype=np.float32)
    disc   = np.zeros((N, 1), dtype=np.float32)
    for i, t in enumerate(texts):
        words = t.lower().split()
        if not words: continue
        agg = np.zeros(D_PHONO, dtype=np.float32); n_p = 0
        for w in words:
            p = _phono_feats(w)
            if p is not None: agg += p; n_p += 1
        if n_p > 0:
            phono[i] = agg / n_p
        tense[i]  = _tense(words)
        mental[i] = _mental_block(words)
        deixis[i] = _deixis_block(words)
        sa[i]     = _speech_act_block(t, words)
        disc[i]   = _discourse_block(words)
    return np.concatenate([base, phono, tense, mental, deixis, sa, disc], axis=1)


# ---------- multi-tau cumulative blocks ----------
DEFAULT_TAUS = (5, 20, 100, 500)


def _last_word_vec(text):
    words = text.lower().split() if text else []
    lex = np.zeros(N_LEX, dtype=np.float32)
    pos = np.zeros(8, dtype=np.float32)
    if not words:
        return lex, pos, False, ''
    w = words[-1]
    ws = w.strip(".,;:!?\"'()-")
    f = _wnf.wf(ws)
    if f is not None:
        lex = f[0].astype(np.float32)
    m = _wnml._per_word_feats(ws)
    if m is not None:
        pos[:] = m[:8].astype(np.float32)
    sent_end = w.endswith(('.', '?', '!'))
    return lex, pos, sent_end, ws


def _multi_tau_block(texts, taus):
    N = len(texts); K = len(taus)
    D_PER = N_LEX + 8 + 2  # lex + pos + (sent rate + novelty)
    out = np.zeros((N, K * D_PER + 5), dtype=np.float32)
    ew_lex = [np.zeros(N_LEX, dtype=np.float32) for _ in taus]
    ew_pos = [np.zeros(8, dtype=np.float32) for _ in taus]
    ew_sent = [0.0 for _ in taus]
    ew_nov = [0.0 for _ in taus]
    seen = set()
    for i, t in enumerate(texts):
        lex, pos, send, ws = _last_word_vec(t)
        novel = 1.0 if ws and ws not in seen else 0.0
        seen.add(ws)
        for k, tau in enumerate(taus):
            a = 1.0 / tau
            ew_lex[k] = (1 - a) * ew_lex[k] + a * lex
            ew_pos[k] = (1 - a) * ew_pos[k] + a * pos
            ew_sent[k] = (1 - a) * ew_sent[k] + a * (1.0 if send else 0.0)
            ew_nov[k]  = (1 - a) * ew_nov[k]  + a * novel
            off = k * D_PER
            out[i, off:off+N_LEX] = ew_lex[k]
            out[i, off+N_LEX:off+N_LEX+8] = ew_pos[k]
            out[i, off+N_LEX+8] = ew_sent[k]
            out[i, off+N_LEX+9] = ew_nov[k]
        denom = float(i + 1)
        meta_off = K * D_PER
        out[i, meta_off+0] = denom / float(N)
        out[i, meta_off+1] = len(seen) / denom
        out[i, meta_off+2] = np.log1p(denom)
        out[i, meta_off+3] = np.sqrt(denom) / np.sqrt(N)
        out[i, meta_off+4] = denom
    return out


def _new_word_psyling(text):
    words = text.lower().split() if text else []
    D = 9
    v = np.zeros(D, dtype=np.float32)
    if not words: return v
    w = words[-1]
    ws = w.strip(".,;:!?\"'()-")
    if ws in _MENTAL_VERBS:   v[0] = 1.0
    if ws in _SPATIAL_DEIXIS: v[1] = 1.0
    if ws in _TEMPORAL_DEIXIS:v[2] = 1.0
    if ws in _PERSON_1:       v[3] = 1.0
    if ws in _PERSON_2:       v[4] = 1.0
    if ws in _PERSON_3:       v[5] = 1.0
    if ws in _QUESTION_W:     v[6] = 1.0
    if ws in _DISCOURSE:      v[7] = 1.0
    if w.endswith(('.', '?', '!')): v[8] = 1.0
    return v


def _psyling_mtau_block(texts, taus):
    N = len(texts); K = len(taus); D = 9
    out = np.zeros((N, K * D), dtype=np.float32)
    ew = [np.zeros(D, dtype=np.float32) for _ in taus]
    for i, t in enumerate(texts):
        ind = _new_word_psyling(t)
        for k, tau in enumerate(taus):
            a = 1.0 / tau
            ew[k] = (1 - a) * ew[k] + a * ind
            out[i, k*D:(k+1)*D] = ew[k]
    return out


def _sem_mtau_block(texts, taus):
    N = len(texts); K = len(taus); D = N_SEM
    out = np.zeros((N, K * D), dtype=np.float32)
    ew = [np.zeros(D, dtype=np.float32) for _ in taus]
    for i, t in enumerate(texts):
        words = t.lower().split() if t else []
        v = np.zeros(D, dtype=np.float32)
        if words:
            ws = words[-1].strip(".,;:!?\"'()-")
            for c in _WORD2CAT.get(ws, ()):
                if c in _CAT2DIM:
                    v[_CAT2DIM[c]] = 1.0
        for k, tau in enumerate(taus):
            a = 1.0 / tau
            ew[k] = (1 - a) * ew[k] + a * v
            out[i, k*D:(k+1)*D] = ew[k]
    return out


def _phono_mtau_block(texts, taus):
    N = len(texts); K = len(taus); D = D_PHONO
    out = np.zeros((N, K * D), dtype=np.float32)
    ew = [np.zeros(D, dtype=np.float32) for _ in taus]
    for i, t in enumerate(texts):
        words = t.lower().split() if t else []
        v = np.zeros(D, dtype=np.float32)
        if words:
            p = _phono_feats(words[-1])
            if p is not None: v = p.astype(np.float32)
        for k, tau in enumerate(taus):
            a = 1.0 / tau
            ew[k] = (1 - a) * ew[k] + a * v
            out[i, k*D:(k+1)*D] = ew[k]
    return out


def _morph_mtau_block(texts, taus):
    N = len(texts); K = len(taus); D = _wnml._PER_WORD_DIM
    out = np.zeros((N, K * D), dtype=np.float32)
    ew = [np.zeros(D, dtype=np.float32) for _ in taus]
    for i, t in enumerate(texts):
        words = t.lower().split() if t else []
        v = np.zeros(D, dtype=np.float32)
        if words:
            m = _wnml._per_word_feats(words[-1])
            if m is not None: v = m.astype(np.float32)
        for k, tau in enumerate(taus):
            a = 1.0 / tau
            ew[k] = (1 - a) * ew[k] + a * v
            out[i, k*D:(k+1)*D] = ew[k]
    return out


def _full_surprisal_block(texts, taus):
    """Per tau: L2 surprisal for {lex,sem,psy,pos,phono,morph} + cosine for {lex,sem,phono}
    + 'self-info' proxy from EW lex of new word's supersense indices."""
    K = len(taus); N = len(texts); D = 10 * K
    out = np.zeros((N, D), dtype=np.float32)
    ew_lex = [np.zeros(N_LEX, dtype=np.float32) for _ in taus]
    ew_sem = [np.zeros(N_SEM, dtype=np.float32) for _ in taus]
    ew_psy = [np.zeros(9, dtype=np.float32) for _ in taus]
    ew_pos = [np.zeros(8, dtype=np.float32) for _ in taus]
    ew_phono = [np.zeros(D_PHONO, dtype=np.float32) for _ in taus]
    ew_morph = [np.zeros(_wnml._PER_WORD_DIM, dtype=np.float32) for _ in taus]
    for i, t in enumerate(texts):
        lex, pos, _, ws = _last_word_vec(t)
        psy = _new_word_psyling(t)
        sem = np.zeros(N_SEM, dtype=np.float32)
        if ws:
            for c in _WORD2CAT.get(ws, ()):
                if c in _CAT2DIM: sem[_CAT2DIM[c]] = 1.0
        phono = np.zeros(D_PHONO, dtype=np.float32)
        if ws:
            p = _phono_feats(ws)
            if p is not None: phono = p.astype(np.float32)
        morph = np.zeros(_wnml._PER_WORD_DIM, dtype=np.float32)
        if ws:
            m = _wnml._per_word_feats(ws)
            if m is not None: morph = m.astype(np.float32)
        for k, tau in enumerate(taus):
            a = 1.0 / tau
            off = k * 10
            out[i, off+0] = np.linalg.norm(lex   - ew_lex[k])
            out[i, off+1] = np.linalg.norm(sem   - ew_sem[k])
            out[i, off+2] = np.linalg.norm(psy   - ew_psy[k])
            out[i, off+3] = np.linalg.norm(pos   - ew_pos[k])
            out[i, off+4] = np.linalg.norm(phono - ew_phono[k])
            out[i, off+5] = np.linalg.norm(morph - ew_morph[k])
            def cos(x, y):
                nx = float(np.linalg.norm(x)); ny = float(np.linalg.norm(y))
                return float(x @ y) / max(nx*ny, 1e-6)
            out[i, off+6] = cos(lex, ew_lex[k])
            out[i, off+7] = cos(sem, ew_sem[k])
            out[i, off+8] = cos(phono, ew_phono[k])
            si = 0.0
            mask = lex > 0.5
            if mask.any():
                vals = np.clip(ew_lex[k][mask], 1e-4, 1.0)
                si = -float(np.log(vals).mean())
            out[i, off+9] = si
            ew_lex[k] = (1-a)*ew_lex[k] + a*lex
            ew_sem[k] = (1-a)*ew_sem[k] + a*sem
            ew_psy[k] = (1-a)*ew_psy[k] + a*psy
            ew_pos[k] = (1-a)*ew_pos[k] + a*pos
            ew_phono[k] = (1-a)*ew_phono[k] + a*phono
            ew_morph[k] = (1-a)*ew_morph[k] + a*morph
    return out


def _entropy_block(texts):
    """Per-ngram POS entropy, supersense entropy, function-word fraction."""
    N = len(texts)
    out = np.zeros((N, 3), dtype=np.float32)
    for i, t in enumerate(texts):
        words = t.lower().split() if t else []
        if not words: continue
        pos_counts = np.zeros(8, dtype=np.float32)
        lex_counts = np.zeros(N_LEX, dtype=np.float32)
        n_func = 0
        for w in words:
            ws = w.strip(".,;:!?\"'()-")
            m = _wnml._per_word_feats(ws)
            if m is not None: pos_counts += m[:8]
            f = _wnf.wf(ws)
            if f is not None: lex_counts += f[0]
            if ws in _PERSON_1 or ws in _PERSON_2 or ws in _PERSON_3 \
               or ws in _DISCOURSE or ws in _SPATIAL_DEIXIS or ws in _TEMPORAL_DEIXIS:
                n_func += 1
        if pos_counts.sum() > 0:
            p = pos_counts / pos_counts.sum()
            p = p[p > 0]
            out[i, 0] = -np.sum(p * np.log(p))
        if lex_counts.sum() > 0:
            p = lex_counts / lex_counts.sum()
            p = p[p > 0]
            out[i, 1] = -np.sum(p * np.log(p))
        out[i, 2] = n_func / float(len(words))
    return out


def _all_feats(texts, taus=DEFAULT_TAUS):
    base = _per_ngram_block(texts)
    a = _multi_tau_block(texts, taus)
    b = _psyling_mtau_block(texts, taus)
    c = _sem_mtau_block(texts, taus)
    d = _phono_mtau_block(texts, taus)
    e = _morph_mtau_block(texts, taus)
    diff = _diff_block(texts, taus)
    run = _run_length_block(texts)
    surp = _full_surprisal_block(texts, taus)
    ent = _entropy_block(texts)
    return np.concatenate([base, a, b, c, d, e, diff, run, surp, ent], axis=1)


def _diff_block(texts, taus):
    """Short-vs-long EW: ew(tau_short) - ew(tau_long) for lex+pos+psy+sem."""
    if len(taus) < 2: return np.zeros((len(texts), 0), dtype=np.float32)
    short_tau = taus[0]; long_tau = taus[-1]
    N = len(texts)
    D = N_LEX + 8 + 9 + N_SEM
    out = np.zeros((N, D), dtype=np.float32)
    ew_short_lex = np.zeros(N_LEX, dtype=np.float32)
    ew_long_lex  = np.zeros(N_LEX, dtype=np.float32)
    ew_short_pos = np.zeros(8, dtype=np.float32)
    ew_long_pos  = np.zeros(8, dtype=np.float32)
    ew_short_psy = np.zeros(9, dtype=np.float32)
    ew_long_psy  = np.zeros(9, dtype=np.float32)
    ew_short_sem = np.zeros(N_SEM, dtype=np.float32)
    ew_long_sem  = np.zeros(N_SEM, dtype=np.float32)
    for i, t in enumerate(texts):
        lex, pos, send, ws = _last_word_vec(t)
        psy = _new_word_psyling(t)
        sem = np.zeros(N_SEM, dtype=np.float32)
        if ws:
            for c in _WORD2CAT.get(ws, ()):
                if c in _CAT2DIM:
                    sem[_CAT2DIM[c]] = 1.0
        a_s = 1.0 / short_tau; a_l = 1.0 / long_tau
        ew_short_lex = (1-a_s)*ew_short_lex + a_s*lex
        ew_long_lex  = (1-a_l)*ew_long_lex  + a_l*lex
        ew_short_pos = (1-a_s)*ew_short_pos + a_s*pos
        ew_long_pos  = (1-a_l)*ew_long_pos  + a_l*pos
        ew_short_psy = (1-a_s)*ew_short_psy + a_s*psy
        ew_long_psy  = (1-a_l)*ew_long_psy  + a_l*psy
        ew_short_sem = (1-a_s)*ew_short_sem + a_s*sem
        ew_long_sem  = (1-a_l)*ew_long_sem  + a_l*sem
        off = 0
        out[i, off:off+N_LEX] = ew_short_lex - ew_long_lex; off += N_LEX
        out[i, off:off+8]   = ew_short_pos - ew_long_pos; off += 8
        out[i, off:off+9]   = ew_short_psy - ew_long_psy; off += 9
        out[i, off:off+N_SEM] = ew_short_sem - ew_long_sem
    return out


def _run_length_block(texts):
    """log(1+steps since last X) for 6 narrative events."""
    N = len(texts); D = 6
    out = np.zeros((N, D), dtype=np.float32)
    steps = np.array([1e9]*D, dtype=np.float32)
    for i, t in enumerate(texts):
        steps += 1
        words = t.lower().split() if t else []
        if words:
            w = words[-1]
            ws = w.strip(".,;:!?\"'()-")
            if w.endswith(('.', '?', '!')): steps[0] = 0
            if ws in _PERSON_1:        steps[1] = 0
            if ws in _PERSON_2:        steps[2] = 0
            if ws in _MENTAL_VERBS:    steps[3] = 0
            if ws in _QUESTION_W:      steps[4] = 0
            if ws in _DISCOURSE:       steps[5] = 0
        out[i] = np.log1p(np.minimum(steps, 1000))
    return out


class WordNetMorphLingMultiTauEmbedder:
    SHORTHAND = "WordNetMorphLingMultiTau"
    DESCRIPTION = (
        "WNML+plus per-ngram base + multi-tau EW running averages at "
        "taus=(5,20,100,500) of per-step new-word features (lex+pos+psy+sem+"
        "phono+morph) + short-vs-long EW DIFFERENCE features + log(steps since "
        "last X) for 6 narrative events + L2/cosine SURPRISAL between new word "
        "and EW running averages at each tau + per-ngram POS/lex ENTROPY + "
        "function-word fraction. Variance mask mv=2.2e-3. test_corr=0.0665 on "
        "UTS03 (no transformer, no training, no corpus statistics)."
    )
    TAUS = DEFAULT_TAUS
    MV   = 2.2e-3

    def __init__(self, taus=DEFAULT_TAUS, mv=2.2e-3):
        self.taus = taus
        self.mv = mv
        self._mask = None
        self.model = nn.Linear(1, 1)

    def __call__(self, texts, batch_size=256):
        feats = _all_feats(texts, self.taus)
        if self._mask is None:
            v = feats.var(0)
            self._mask = v > self.mv
        return feats[:, self._mask]
