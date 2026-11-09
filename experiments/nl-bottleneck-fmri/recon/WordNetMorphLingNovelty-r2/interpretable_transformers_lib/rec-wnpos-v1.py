"""Hand-written text features for predicting fMRI responses. Edit this file.

build_embedder() must return an object that maps a list of 10-gram strings (all of one story's
10-grams, in order) to a float array of shape (number of 10-grams, number of features).

No training. Position features (story / sentence / paragraph units, punctuation decays) plus
WordNet lexname content, semantic-category table, word lists, suffixes, letter phonology,
summed over the 10-gram and running averages, plus novelty features.
"""
import re
import numpy as np

MODEL_NAME = 'rec-wnpos-v1'
MODEL_DESCRIPTION = ('Reconstruction: story-position Fourier/Chebyshev/Legendre/Gaussian bumps/log-edge, '
                     'sentence+paragraph unit features, punctuation decays; WordNet lexnames direct+expanded, '
                     '72 anchor-synset categories, word lists, suffixes, letter phonology; 10-gram sums and '
                     'EMAs tau 5-500; novelty; drop var<0.05 on first story.')

try:
    from nltk.corpus import wordnet as wn
    wn.synsets('dog')
except Exception:  # pragma: no cover
    wn = None

LEXNAMES = ['adj.all', 'adj.pert', 'adj.ppl', 'adv.all', 'noun.Tops', 'noun.act', 'noun.animal',
            'noun.artifact', 'noun.attribute', 'noun.body', 'noun.cognition', 'noun.communication',
            'noun.event', 'noun.feeling', 'noun.food', 'noun.group', 'noun.location', 'noun.motive',
            'noun.object', 'noun.person', 'noun.phenomenon', 'noun.plant', 'noun.possession',
            'noun.process', 'noun.quantity', 'noun.relation', 'noun.shape', 'noun.state',
            'noun.substance', 'noun.time', 'verb.body', 'verb.change', 'verb.cognition',
            'verb.communication', 'verb.competition', 'verb.consumption', 'verb.contact',
            'verb.creation', 'verb.emotion', 'verb.motion', 'verb.perception', 'verb.possession',
            'verb.social', 'verb.stative', 'verb.weather']
LEX_IDX = {n: i for i, n in enumerate(LEXNAMES)}

# 72 semantic categories defined by anchor synsets (membership = hypernym closure contains anchor)
ANCHORS = [
    'person.n.01', 'relative.n.01', 'female.n.02', 'male.n.02', 'child.n.01', 'adult.n.01',
    'leader.n.01', 'worker.n.01', 'body_part.n.01', 'animal.n.01', 'plant.n.02', 'food.n.01',
    'beverage.n.01', 'building.n.01', 'room.n.01', 'vehicle.n.01', 'furniture.n.01',
    'clothing.n.01', 'tool.n.01', 'weapon.n.01', 'device.n.01', 'container.n.01',
    'location.n.01', 'region.n.03', 'body_of_water.n.01', 'geological_formation.n.01',
    'time_period.n.01', 'time_unit.n.01', 'event.n.01', 'emotion.n.01', 'feeling.n.01',
    'cognition.n.01', 'communication.n.02', 'language.n.01', 'written_communication.n.01',
    'music.n.01', 'money.n.01', 'possession.n.02', 'organization.n.01', 'social_group.n.01',
    'number.n.02', 'measure.n.02', 'color.n.01', 'shape.n.02', 'substance.n.01',
    'material.n.01', 'natural_phenomenon.n.01', 'weather.n.01', 'disease.n.01', 'act.n.02',
    'activity.n.01', 'sport.n.01', 'game.n.01', 'sound.n.01', 'light.n.01', 'artifact.n.01',
    'structure.n.01', 'document.n.01', 'travel.v.01', 'communicate.v.01', 'think.v.03',
    'perceive.v.01', 'see.v.01', 'hear.v.01', 'feel.v.01', 'eat.v.01', 'kill.v.01',
    'get.v.01', 'make.v.03', 'change.v.01', 'touch.v.01', 'love.v.01',
]

WORDLISTS = {
    'pron1s': 'i me my mine myself',
    'pron1p': 'we us our ours ourselves',
    'pron2': 'you your yours yourself yourselves',
    'pron3m': 'he him his himself',
    'pron3f': 'she her hers herself',
    'pron3n': 'it its itself',
    'pron3p': 'they them their theirs themselves',
    'demo': 'this that these those',
    'indef': 'someone somebody something anyone anybody anything everyone everybody everything nobody nothing none',
    'modal': 'can could may might must shall should will would ought',
    'neg': "not no never nothing nobody none neither nor cannot n't don't didn't won't can't isn't wasn't aren't weren't doesn't couldn't wouldn't shouldn't haven't hasn't hadn't",
    'aux': 'be am is are was were been being have has had having do does did',
    'det': 'the a an some any every each all both either',
    'conj': 'and or but so because although though if when while since until unless whereas',
    'prep': 'in on at to from with without by for of about over under into onto through across between among after before during against around behind',
    'wh': 'what who whom whose which where when why how',
    'quant': 'many much more most few less least several lot lots',
    'time': 'now then today yesterday tomorrow later soon ago already still yet always often sometimes',
    'space': 'here there up down out away back home inside outside',
    'filler': 'um uh like yeah okay oh well',
    'intens': 'very really so too quite pretty just totally',
    'say': 'said say says saying told tell tells asked ask',
}
WL = {k: set(v.split()) for k, v in WORDLISTS.items()}
STOP = set().union(*WL.values()) | {'as', 'than', 'also', 'only', 'not', 'into', 'like'}

SUFFIXES = ['ing', 'ed', 'ly', 's', 'es', 'tion', 'sion', 'ness', 'ment', 'er', 'est', 'able',
            'ible', 'ful', 'less', 'ize', 'al', 'ity', 'ous', 'ive', 'ic', 'y', "'s", "n't"]
VOWELS = set('aeiouy')


def clean(w):
    w = w.lower()
    return re.sub(r"[^a-z']", '', w).strip("'") or w


_cache = {}
_anchor_syn = None


def _anchors():
    global _anchor_syn
    if _anchor_syn is None:
        _anchor_syn = []
        for a in ANCHORS:
            try:
                _anchor_syn.append(wn.synset(a) if wn else None)
            except Exception:
                _anchor_syn.append(None)
    return _anchor_syn


def _lex_direct(syns):
    v = np.zeros(len(LEXNAMES))
    for s in syns:
        i = LEX_IDX.get(s.lexname())
        if i is not None:
            v[i] += 1
    return v / v.sum() if v.sum() > 0 else v


def word_feats(w):
    if w in _cache:
        return _cache[w]
    L = len(LEXNAMES)
    direct = np.zeros(L); expanded = np.zeros(L); cats = np.zeros(len(ANCHORS))
    if wn is not None and w not in STOP and len(w) > 1:
        syns = wn.synsets(w)[:5]
        if syns:
            direct = _lex_direct(syns)
            hyp = []
            gl = []
            for s in syns[:3]:
                hyp += s.hypernyms()
                for g in re.findall(r'[a-z]+', s.definition().lower()):
                    if len(g) > 3 and g not in STOP:
                        gs = wn.synsets(g)
                        if gs:
                            gl.append(gs[0])
            expanded = direct + 0.5 * _lex_direct(hyp) + 0.5 * _lex_direct(gl)
            expanded = expanded / expanded.sum() if expanded.sum() > 0 else expanded
            closure = set()
            for s in syns[:3]:
                closure.add(s)
                closure |= set(s.closure(lambda x: x.hypernyms() + x.instance_hypernyms()))
            for j, a in enumerate(_anchors()):
                if a is not None and a in closure:
                    cats[j] = 1.0
    lists = np.array([1.0 if w in WL[k] else 0.0 for k in WL])
    suf = np.array([1.0 if (w.endswith(s) and len(w) > len(s) + 2) else 0.0 for s in SUFFIXES])
    nv = sum(c in VOWELS for c in w)
    syl = max(1, len(re.findall(r'[aeiouy]+', w)))
    clusters = len(re.findall(r'[^aeiouy\']{2,}', w))
    phon = np.array([len(w), np.log1p(len(w)), syl, nv / max(1, len(w)), clusters,
                     float(w[:1] in VOWELS), float(w[-1:] in VOWELS), float(w in STOP),
                     float(any(w.startswith(c) for c in ('th', 'sh', 'ch', 'wh')))])
    f = np.concatenate([direct, expanded, cats, lists, suf, phon])
    _cache[w] = f
    return f


def scale05(M):
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        M = M[:, None]
    M = M - M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-9] = 1.0
    return M / sd * np.sqrt(0.5)


def position_feats(n):
    idx = np.arange(n)
    p = (idx + 0.5) / n
    cols = []
    for k in range(1, 11):
        cols += [np.sin(2 * np.pi * k * p), np.cos(2 * np.pi * k * p)]
    x = 2 * p - 1
    for d in range(1, 11):
        c = np.zeros(d + 1); c[d] = 1
        cols.append(np.polynomial.chebyshev.chebval(x, c))
        cols.append(np.polynomial.legendre.legval(x, c))
    # ~65 gaussian bumps
    for width, nc in [(0.03, 33), (0.07, 17), (0.15, 9), (0.3, 5), (0.6, 1)]:
        centers = np.linspace(0, 1, nc) if nc > 1 else np.array([0.5])
        for c in centers:
            cols.append(np.exp(-0.5 * ((p - c) / width) ** 2))
    cols.append(np.log1p(idx))
    cols.append(np.log1p(n - 1 - idx))
    return scale05(np.stack(cols, 1))


def unit_feats(unit_bounds, n):
    """unit_bounds: list of (start, end_exclusive)."""
    F = np.zeros((n, 10))
    for s, e in unit_bounds:
        L = e - s
        j = np.arange(L)
        r = (j + 0.5) / L
        F[s:e, 0] = r
        F[s:e, 1] = np.log1p(j)
        F[s:e, 2] = np.log1p(L - 1 - j)
        F[s:e, 3] = np.log(L)
        F[s, 4] = 1
        F[e - 1, 5] = 1
        F[s:e, 6] = np.sin(2 * np.pi * r); F[s:e, 7] = np.cos(2 * np.pi * r)
        F[s:e, 8] = np.sin(4 * np.pi * r); F[s:e, 9] = np.cos(4 * np.pi * r)
    return F


def decay_feats(flags, n, taus):
    """flags: bool array of event at word. since-last and until-next decays."""
    since = np.full(n, np.inf); until = np.full(n, np.inf)
    last = None
    for i in range(n):
        if last is not None:
            since[i] = i - last
        if flags[i]:
            last = i
    nxt = None
    for i in range(n - 1, -1, -1):
        if flags[i]:
            nxt = i
        if nxt is not None:
            until[i] = nxt - i
    cols = []
    for t in taus:
        cols.append(np.exp(-since / t)); cols.append(np.exp(-until / t))
    return np.stack(cols, 1)


class Embedder:
    def __init__(self):
        self.keep = None

    def __call__(self, texts):
        n = len(texts)
        raw = [(t.split()[-1] if t.split() else '') for t in texts]
        words = [clean(w) for w in raw]
        blocks = []
        # --- story position
        blocks.append(position_feats(n))
        # --- sentences and paragraphs
        stripped = [re.sub(r'["\')\]]+$', '', w) for w in raw]
        end_any = np.array([w.endswith(('.', '?', '!')) for w in stripped])
        sents = []
        s = 0
        for i in range(n):
            if end_any[i] or i == n - 1:
                sents.append((s, i + 1)); s = i + 1
        units = [sents]
        for Lmin in [10, 20, 40, 80, 160, 320]:
            paras = []; cur = None
            for (a, b) in sents:
                cur = (a, b) if cur is None else (cur[0], b)
                if cur[1] - cur[0] >= Lmin:
                    paras.append(cur); cur = None
            if cur is not None:
                if paras:
                    paras[-1] = (paras[-1][0], cur[1])
                else:
                    paras.append(cur)
            units.append(paras)
        for u in units:
            blocks.append(unit_feats(u, n))
        taus = [2, 5, 10, 20, 50, 100, 200]
        blocks.append(decay_feats(end_any, n, taus))
        for ch in '.?!':
            blocks.append(decay_feats(np.array([w.endswith(ch) for w in stripped]), n, taus))
        # --- content
        W = np.stack([word_feats(w) for w in words], 0)
        cs = np.vstack([np.zeros((1, W.shape[1])), np.cumsum(W, 0)])
        lo = np.maximum(np.arange(n) - 9, 0)
        blocks.append(cs[np.arange(n) + 1] - cs[lo])  # 10-gram sum
        for tau in [5, 20, 100, 500]:
            a = 1.0 / tau
            E = np.zeros_like(W); acc = np.zeros(W.shape[1])
            for i in range(n):
                acc = (1 - a) * acc + a * W[i]
                E[i] = acc
            blocks.append(E)
        # --- novelty
        nov = np.zeros((n, 9))
        lastpos = {}; counts = {}
        types = 0
        for i, w in enumerate(words):
            first = w not in lastpos
            if first:
                types += 1
            gap = i - lastpos[w] if not first else i + 1
            counts[w] = counts.get(w, 0) + 1
            win = words[max(0, i - 9):i + 1]
            w50 = words[max(0, i - 49):i + 1]
            w200 = words[max(0, i - 199):i + 1]
            nov[i] = [float(first), np.log1p(gap), types / (i + 1), len(set(win)) / len(win),
                      len(set(w50)) / len(w50), len(set(w200)) / len(w200),
                      np.log1p(counts[w]), np.log1p(types), float(first and w not in STOP)]
            lastpos[w] = i
        blocks.append(nov)
        X = np.concatenate(blocks, 1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.keep is None:
            self.keep = X.var(0) >= 0.05
        return X[:, self.keep].astype(np.float32)


def build_embedder():
    return Embedder()
