"""WordNet + morph + ling + phono + tense + mentalizing + deixis + speech-act +
discourse + variance mask. test_corr=0.0610 on UTS03 (UTS03, num_train=8, num_test=3).

A ~0.0018 absolute improvement over WordNetMorphLing-tau50 (0.0592) achieved by:
  1. Adding 10 phonological surface features (initial cluster, double-consonant
     ending, stop/fricative/nasal/liquid density, 'th' count, vowel ending,
     repeated vowels, consonant clusters). Captures sound-level signal that the
     auditory cortex picks up (the Huth fMRI stimuli are spoken).
  2. Adding 5 tense ratio features (V-ed, V-ing, past/present/future auxiliary
     densities). Tense isolates a fast-changing temporal axis that the
     morphological -ing/-ed flags don't capture in ratio form.
  3. Adding 5 mentalizing density features (mental-state verbs, perception
     verbs, emotion words, social relation nouns, speech-act verbs). These
     target TPJ/mPFC theory-of-mind regions.
  4. Adding 6 deixis density features (spatial, temporal, 1st/2nd/3rd/we
     person markers). Deixis anchors discourse to speaker/listener context.
  5. Adding 4 speech-act features (question word density, '?' present,
     '!' present, imperative-hint density). Encodes utterance type.
  6. Adding 1 discourse-marker density feature (and/but/so/then/...).
  7. **Variance mask (mv=1e-4)** computed on the FIRST training story's
     ngrams. Drops dims with var<1e-4 in that ~1700-ngram pool. Roughly 50
     mostly-zero dims are eliminated, reducing FIR-expansion noise that
     ridge can't shrink away (alpha grid is logspace(1,4,12)).

Architecture (per 10-gram window):
  - WordNetMorphLing-tau50 base = 406 dims
  - + phono mean = 10 dims
  - + tense ratios = 5 dims
  - + mentalizing = 5 dims, deixis = 6, speech_act = 4, discourse = 1
  - Total raw = 437 dims
  - Variance mask (on first train story) keeps ~384 dims
  - FIR-delayed (x4) for ridge: ~1536 dims at TR-level

Key findings:
  - The mask is the bigger lift (+0.0009 alone). The tiny semantic blocks
    (phono+tense+mental+deixis+speech+discourse) contribute the remaining
    +0.0009.
  - Each addition individually contributes 0.0001-0.0003. Adding too many
    blocks (>16 total) starts to hurt — ridge over-distributes weights.
  - Variance mv=1e-4 is the sweet spot. Lower keeps too many zero-var dims;
    higher drops useful low-var signal.
  - Cross-story mask (computed from all 8 train stories) is WORSE than
    single-story mask, presumably because dims missing from story 1 are
    rarely useful in test stories.
  - WN supersense bigrams (2025 dims) add nothing beyond the mask plateau.
  - Position-replicated word blocks (2nd-to-last, first word) overfit
    (train_corr ↑ but test_corr ↓).

What did NOT help on top of this base:
  - More compositional bigram features (lex_bigram, cat_bigram, skip-bigram)
  - Position-resolved word blocks
  - Concreteness/informativeness via WN depth
  - Animacy splits (human/animal/object/abstract)
  - Hedges/modality
  - WN-anchor-distance, sensorimotor diffusion, SVO roles, WSD context

Gap to GPT-2 XL baseline (0.0791): 0.018, still substantial.
"""
import numpy as np
import torch.nn as nn
import os
import importlib.util
from functools import lru_cache


_HERE = os.path.dirname(os.path.abspath(__file__))

# Reuse WordNetMorphLing-tau50 base (D=406)
_wnml_spec = importlib.util.spec_from_file_location(
    "_wnml_base", os.path.join(_HERE, "WordNetMorphLing-tau50.py"))
_wnml = importlib.util.module_from_spec(_wnml_spec); _wnml_spec.loader.exec_module(_wnml)


# ============================================================================
# Phonological surface features (per word; aggregated by mean over window).
# 10 features capture sound-level structure that auditory cortex picks up.
# ============================================================================
import re
from functools import lru_cache as _lru_cache

_STOPS = set('ptkbdg')
_FRICS = set('fvsz')  # plus 'th' digraph counted separately as v[6]
_NASALS = set('mn')
_LIQUIDS = set('lr')
_INITIAL_CLUSTERS = ['st','sp','sk','pl','pr','tr','br','gr','cr','dr',
                     'fl','fr','sl','sm','sn','sw','tw','th','sh','ch','wh']

D_PHONO = 10


@_lru_cache(maxsize=200000)
def _phono_feats(word):
    w = word.strip(".,;:!?\"'()-").lower()
    if not w:
        return None
    L = len(w)
    v = np.zeros(D_PHONO, dtype=np.float32)
    # 0: initial cluster
    v[0] = 1.0 if any(w.startswith(c) for c in _INITIAL_CLUSTERS) else 0.0
    # 1: double-letter consonant ending (long-vowel marker, e.g. 'ss', 'tt')
    v[1] = 1.0 if (L >= 2 and w[-1] == w[-2] and w[-1] not in 'aeiou') else 0.0
    # 2-5: phoneme class densities
    v[2] = sum(1 for c in w if c in _STOPS) / L
    v[3] = sum(1 for c in w if c in _FRICS) / L
    v[4] = sum(1 for c in w if c in _NASALS) / L
    v[5] = sum(1 for c in w if c in _LIQUIDS) / L
    # 6: 'th' digraph
    v[6] = w.count('th') / L
    # 7: ends in vowel (open syllable)
    v[7] = 1.0 if w[-1] in 'aeiou' else 0.0
    # 8: repeated vowels (long vowel marker)
    v[8] = 1.0 if re.search(r'[aeiou]{2,}', w) else 0.0
    # 9: consonant cluster count
    v[9] = len(re.findall(r'[^aeiouy]{2,}', w)) / L
    return v


# ============================================================================
# Tense ratio features (5 dims).
# ============================================================================
_PAST_AUX = {'was','were','had','did'}
_PRES_AUX = {'is','are','am','have','has','do','does'}
_FUT_AUX = {'will','shall',"'ll"}


def _tense_feats(words):
    v = np.zeros(5, dtype=np.float32)
    L = max(len(words), 1)
    for w in words:
        wl = w.strip(".,;:!?\"'()-").lower()
        if len(wl) > 3 and wl.endswith('ed'): v[0] += 1   # V-ed
        if len(wl) > 4 and wl.endswith('ing'): v[1] += 1  # V-ing
        if wl in _PAST_AUX: v[2] += 1
        if wl in _PRES_AUX: v[3] += 1
        if wl in _FUT_AUX: v[4] += 1
    return v / L


# ============================================================================
# Mentalizing / social density (5 dims). Targets TPJ/mPFC.
# ============================================================================
_MENTAL_VERBS = {
    'think','thinks','thought','thinking',
    'know','knows','knew','known','knowing',
    'believe','believes','believed','believing',
    'hope','hopes','hoped','hoping',
    'want','wants','wanted','wanting',
    'wish','wishes','wished','wishing',
    'feel','feels','felt','feeling',
    'remember','remembers','remembered','remembering',
    'forget','forgets','forgot','forgetting',
    'understand','understands','understood',
    'realize','realizes','realized','realizing',
    'imagine','imagines','imagined','imagining',
    'wonder','wonders','wondered','wondering',
    'suppose','guess','expect','doubt',
    'decide','decides','decided','deciding',
    'mean','means','meant',
}
_PERCEPT_VERBS = {
    'see','sees','saw','seen','seeing',
    'hear','hears','heard','hearing',
    'look','looks','looked','looking',
    'watch','watches','watched','watching',
    'listen','listens','listened',
    'notice','notices','noticed',
    'observe','observes','observed',
}
_EMOTION_WORDS = {
    'happy','sad','angry','scared','afraid','fear','fearful',
    'joy','joyful','love','hate','glad','sorry',
    'excited','worried','nervous','calm','upset',
    'mad','frustrated','pleased','delighted','depressed',
    'anxious','embarrassed','proud','ashamed','guilty',
    'lonely','jealous','surprised','shocked','disappointed',
}
_SOCIAL_REL = {
    'friend','friends','enemy','enemies',
    'mother','mom','mama','father','dad','papa',
    'sister','brother','son','daughter','child','children','kid','kids','baby',
    'husband','wife','partner','lover',
    'aunt','uncle','cousin','grandfather','grandmother','grandma','grandpa',
    'neighbor','colleague','boss','teacher','student',
    'family','parent','parents','relative','relatives',
}
_SPEECH_VERBS = {
    'say','says','said','saying',
    'tell','tells','told','telling',
    'ask','asks','asked','asking',
    'answer','answers','answered',
    'reply','replies','replied',
    'speak','speaks','spoke','spoken','speaking',
    'talk','talks','talked','talking',
    'whisper','whispers','whispered',
    'shout','shouts','shouted','shouting',
    'yell','yells','yelled','yelling',
    'mention','mentions','mentioned',
    'explain','explains','explained',
}


def _mental_feats(words):
    v = np.zeros(5, dtype=np.float32)
    L = max(len(words), 1)
    for w in words:
        wl = w.strip(".,;:!?\"'()-").lower()
        if wl in _MENTAL_VERBS: v[0] += 1
        if wl in _PERCEPT_VERBS: v[1] += 1
        if wl in _EMOTION_WORDS: v[2] += 1
        if wl in _SOCIAL_REL: v[3] += 1
        if wl in _SPEECH_VERBS: v[4] += 1
    return v / L


# ============================================================================
# Deixis (6 dims).
# ============================================================================
_SPATIAL_DEIXIS = {'here','there','this','that','these','those'}
_TEMPORAL_DEIXIS = {'now','then','today','yesterday','tomorrow','soon','later','earlier','recently','currently'}
_PERSON_1 = {'i',"i'm","i've","i'd","i'll",'me','my','mine','myself'}
_PERSON_2 = {'you',"you're","you've","you'd","you'll",'your','yours','yourself','yourselves'}
_PERSON_3 = {'he','she','it','they','him','her','them','his','hers','its','their','theirs','himself','herself','itself','themselves'}
_PERSON_1PL = {'we',"we're","we've","we'd","we'll",'us','our','ours','ourselves'}


def _deixis_feats(words):
    v = np.zeros(6, dtype=np.float32)
    L = max(len(words), 1)
    for w in words:
        wl = w.strip(".,;:!?\"'()-").lower()
        if wl in _SPATIAL_DEIXIS: v[0] += 1
        if wl in _TEMPORAL_DEIXIS: v[1] += 1
        if wl in _PERSON_1: v[2] += 1
        if wl in _PERSON_2: v[3] += 1
        if wl in _PERSON_3: v[4] += 1
        if wl in _PERSON_1PL: v[5] += 1
    return v / L


# ============================================================================
# Speech-act (4 dims).
# ============================================================================
_QUESTION_W = {'what','when','where','why','how','who','whom','whose','which'}
_IMPER_HINTS = {"don't",'do','please','just','try','let'}


def _speech_act_feats(text, words):
    v = np.zeros(4, dtype=np.float32)
    L = max(len(words), 1)
    for w in words:
        wl = w.strip(".,;:!?\"'()-").lower()
        if wl in _QUESTION_W: v[0] += 1
        if wl in _IMPER_HINTS: v[3] += 1
    v[0] /= L
    v[3] /= L
    v[1] = 1.0 if '?' in text else 0.0
    v[2] = 1.0 if '!' in text else 0.0
    return v


# ============================================================================
# Discourse marker density (1 dim).
# ============================================================================
_DISCOURSE = {'and','but','so','then','because','however','although','though',
              'meanwhile','suddenly','finally','first','next','after','before',
              'while','when','since','until','unless','if'}


def _discourse_feats(words):
    v = np.zeros(1, dtype=np.float32)
    L = max(len(words), 1)
    for w in words:
        wl = w.strip(".,;:!?\"'()-").lower()
        if wl in _DISCOURSE: v[0] += 1
    return v / L


# ============================================================================
# Combine all features per ngram-window text.
# ============================================================================
def _feats(texts):
    """Returns (N, 437) raw feature matrix."""
    base = _wnml._feats(texts, (50.0,))  # (N, 406)
    N = len(texts)

    # phono mean
    phono = np.zeros((N, D_PHONO), dtype=np.float32)
    for i, t in enumerate(texts):
        words = t.lower().split()
        if not words: continue
        agg = np.zeros(D_PHONO, dtype=np.float32); n_p = 0
        for w in words:
            p = _phono_feats(w)
            if p is not None: agg += p; n_p += 1
        if n_p > 0:
            phono[i] = agg / n_p

    # tense
    tense = np.zeros((N, 5), dtype=np.float32)
    for i, t in enumerate(texts):
        words = t.lower().split()
        if words:
            tense[i] = _tense_feats(words)

    # mental + deixis + speech_act + discourse
    md = np.zeros((N, 5+6+4+1), dtype=np.float32)
    for i, t in enumerate(texts):
        words = t.lower().split()
        if not words: continue
        md[i, 0:5] = _mental_feats(words)
        md[i, 5:11] = _deixis_feats(words)
        md[i, 11:15] = _speech_act_feats(t, words)
        md[i, 15:16] = _discourse_feats(words)

    return np.concatenate([base, phono, tense, md], axis=1)


# ============================================================================
# Embedder (with variance mask).
# ============================================================================
class WNMLPlusMaskedEmbedder:
    """Pure-feature embedder. WordNetMorphLing + phono + tense + mentalizing +
    deixis + speech-act + discourse + variance mask. Returns (N, ~384)."""

    SHORTHAND = "WordNetMorphLingPlusMasked"
    DESCRIPTION = (
        "WordNetMorphLing-tau50 base (D=406) + 10 phonological + 5 tense + "
        "5 mentalizing + 6 deixis + 4 speech-act + 1 discourse = 437 raw dims. "
        "Variance mask mv=1e-4 computed on first train story keeps ~384 dims. "
        "test_corr=0.0610 on UTS03. Pure hand-crafted features, no training, "
        "no pretrained weights, no corpus statistics."
    )

    def __init__(self, min_var: float = 1e-4):
        self.min_var = min_var
        self._mask = None
        self.model = nn.Linear(1, 1)  # placeholder so framework's .to(device) works

    def __call__(self, texts, batch_size=256):
        feats = _feats(texts)
        if self._mask is None:
            v = feats.var(0)
            self._mask = v > self.min_var
        return feats[:, self._mask]
