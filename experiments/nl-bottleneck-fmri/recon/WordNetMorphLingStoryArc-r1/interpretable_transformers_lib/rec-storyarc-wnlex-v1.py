"""Hand-written text features for predicting fMRI responses.

Story-arc (position) block + lexical content block (WordNet lexnames, curated semantic
categories, function words, closed-class lists, suffix/length/syllable cues), summed over the
10-gram, kept for the newest word, exponentially-decayed story averages, differences, and timing
counters. Dimensions with variance <= 0.14 on the first story seen are dropped.
"""
import re
import numpy as np

MODEL_NAME = 'rec-storyarc-wnlex-v1'
MODEL_DESCRIPTION = ('Story-arc position block (108 dims, var 0.5) + WordNet lexname/72 semantic cats/'
                     '50 function words/closed class/suffix/length cues: 10-gram sum, newest word, '
                     'decayed averages (5,20,100,500), diffs, distances, timing counters; '
                     'variance mask <=0.14 on first story.')

try:
    from nltk.corpus import wordnet as _wn
    _wn.synsets('dog')
except Exception:  # pragma: no cover
    _wn = None

LEXNAMES = ('adj.all adj.pert adj.ppl adv.all noun.Tops noun.act noun.animal noun.artifact '
            'noun.attribute noun.body noun.cognition noun.communication noun.event noun.feeling '
            'noun.food noun.group noun.location noun.motive noun.object noun.person '
            'noun.phenomenon noun.plant noun.possession noun.process noun.quantity noun.relation '
            'noun.shape noun.state noun.substance noun.time verb.body verb.change verb.cognition '
            'verb.communication verb.competition verb.consumption verb.contact verb.creation '
            'verb.emotion verb.motion verb.perception verb.possession verb.social verb.stative '
            'verb.weather').split()
LEX_IDX = {n: i for i, n in enumerate(LEXNAMES)}

SEMANTIC = {
    'family': 'mother father mom dad mommy daddy parent sister brother son daughter family wife husband aunt uncle cousin grandmother grandfather grandma grandpa baby kid child children',
    'friend': 'friend buddy pal roommate partner neighbor colleague classmate companion',
    'romance': 'love lover kiss date dating boyfriend girlfriend romantic marry married marriage wedding sex sexy crush',
    'person': 'man woman guy girl boy person people lady gentleman someone somebody everybody everyone human folks',
    'male': 'he him his himself man men boy guy father son brother husband king mister mr',
    'female': 'she her hers herself woman women girl lady mother daughter sister wife queen mrs ms',
    'body': 'body head face eye eyes hand hands arm arms leg legs foot feet hair mouth nose ear skin heart back neck finger shoulder chest stomach blood bone',
    'health': 'sick ill hospital doctor nurse medicine disease cancer pain hurt injury surgery patient health healthy pill drug therapy',
    'death': 'die died dead death kill killed funeral grave murder suicide corpse buried bury',
    'violence': 'fight hit punch shoot shot gun knife attack beat blood war soldier weapon bomb violent kick',
    'crime_law': 'police cop arrest jail prison crime criminal law lawyer court judge steal stole thief illegal guilty',
    'positive_emo': 'happy joy glad love laugh smile excited fun great wonderful amazing proud hope beautiful nice good',
    'negative_emo': 'sad cry tears upset angry mad hate afraid scared fear worried anxious terrible awful bad horrible lonely depressed',
    'fear': 'afraid scared fear terrified panic nervous anxious worry worried frightened dread',
    'anger': 'angry mad furious rage hate annoyed yell scream shout pissed',
    'sadness': 'sad cry cried tears grief lonely miss depressed sorrow heartbroken',
    'surprise': 'surprise surprised shock shocked suddenly amazed wow unexpected',
    'communication': 'say said tell told ask asked talk speak spoke call called answer reply explain shout whisper mention conversation',
    'speech_noun': 'word words story question answer letter email message phone call voice language name',
    'cognition': 'think thought know knew believe understand remember forget realize wonder imagine decide idea mind guess figure',
    'perception': 'see saw look watch hear heard listen feel felt smell taste notice',
    'vision': 'see saw look looked watch eye eyes light dark color bright picture sight view',
    'sound': 'sound noise hear heard listen loud quiet silence music voice song ring',
    'touch': 'touch hold grab push pull hug press rub feel soft hard',
    'motion': 'go went come came walk walked run ran move drive drove fly flew climb jump fall fell leave left arrive return',
    'posture': 'sit sat stand stood lie lay sleep slept wake woke rest',
    'spatial': 'up down in out over under above below behind front inside outside around through across between near far top bottom side',
    'place': 'place room house home city town street road country world school office building park floor',
    'home': 'home house room kitchen bed bedroom bathroom door window table couch apartment wall floor',
    'building': 'building church hospital school hotel restaurant store shop bar office station tower',
    'geography': 'america york california chicago texas london paris china africa europe state country city village island mountain',
    'nature': 'tree trees forest grass flower river lake ocean sea mountain sky sun moon star earth field rock',
    'weather': 'rain snow wind storm cold hot warm weather cloud sunny ice winter summer',
    'water': 'water river lake ocean sea swim wet boat beach wave rain drink',
    'animal': 'dog cat horse bird fish animal cow pig mouse snake bear wolf lion chicken',
    'food': 'food eat ate dinner lunch breakfast meal bread meat cook cooking kitchen restaurant hungry pizza coffee',
    'drink': 'drink drank water beer wine coffee tea alcohol bottle glass drunk bar',
    'clothing': 'shirt dress shoes coat hat jacket pants clothes wear wore suit',
    'vehicle': 'car truck bus train plane bike boat ship taxi drive drove ride',
    'travel': 'trip travel journey flight airport hotel visit vacation road map',
    'time_unit': 'time day days night week weeks month months year years hour hours minute minutes second morning evening afternoon',
    'temporal_adv': 'now then later soon before after already still again always never sometimes once yesterday today tomorrow finally',
    'age': 'old young age older younger kid child adult teenager baby born birthday',
    'number': 'one two three four five six seven eight nine ten hundred thousand million first second third twenty',
    'quantity': 'all some many much more most few little lot lots every each several enough half whole none',
    'size': 'big small large huge tiny little giant tall short long wide',
    'money': 'money pay paid dollar dollars cash buy bought sell sold cost price rich poor job bank',
    'work': 'work job boss office company business career employee hire fired meeting project worked working',
    'school': 'school class teacher student college university study learn teach homework professor grade',
    'science_tech': 'computer phone internet science research data machine technology system scientist experiment',
    'politics': 'government president politics election vote law power state nation political party',
    'war': 'war army soldier military battle enemy fight gun bomb troops',
    'religion': 'god church pray prayer jesus heaven hell religious faith holy soul spirit',
    'music_art': 'music song sing band play guitar piano dance art paint painting movie film show',
    'sport_game': 'game play played team ball win won lose lost score sport player match',
    'color': 'red blue green yellow black white color brown pink purple gray',
    'light': 'light dark bright sun shine lamp shadow night fire',
    'object': 'thing things box bag book paper table chair key bottle stuff piece',
    'possession': 'have has had own get got give gave take took keep kept lose lost',
    'creation': 'make made build built create write wrote draw design fix',
    'change': 'change changed become became turn turned grow grew start started stop stopped begin began end ended',
    'social_event': 'party wedding meeting dinner date visit celebration show',
    'help_care': 'help helped care support save protect thank thanks please kind',
    'conflict': 'fight argue argument problem trouble wrong blame against',
    'truth': 'true truth real really honest lie lied fact actually',
    'certainty': 'sure certain definitely absolutely obviously clearly exactly',
    'uncertainty': 'maybe perhaps probably might possibly guess seem seemed whether',
    'desire': 'want wanted wish need needed hope desire try tried',
    'causation': 'because why reason cause so therefore since result',
    'evaluation': 'good bad better best worse worst right wrong important great fine',
    'intensity': 'very really so too totally completely incredibly extremely quite',
    'sleep_dream': 'sleep slept dream dreamed bed tired awake night',
}
SEM_NAMES = list(SEMANTIC)

FUNCTION_WORDS = ('the a an and or but of to in on at for with from by as that this it is was be '
                  'have had not i you he she we they me my his her so if then when what there all '
                  'just like up out about would do one').split()
FW_IDX = {w: i for i, w in enumerate(FUNCTION_WORDS)}

CLOSED = {
    'pron_1sg': 'i me my mine myself',
    'pron_1pl': 'we us our ours ourselves',
    'pron_2': 'you your yours yourself yourselves',
    'pron_3sg': 'he him his himself she her hers herself it its itself',
    'pron_3pl': 'they them their theirs themselves',
    'modal': 'can could will would shall should may might must',
    'negation': "not no never nothing nobody none nowhere neither nor n't don't didn't can't won't wasn't isn't couldn't wouldn't",
    'mental_verb': 'think thought know knew believe remember forget realize understand wonder feel felt guess suppose imagine',
    'deixis': 'here there now then this that these those today yesterday tomorrow',
    'discourse': 'well so like um uh okay ok yeah oh anyway actually right basically',
    'conjunction': 'and but or because although though while if unless until since whereas',
    'preposition': 'of in on at to for with from by about into over under through after before between against',
    'determiner': 'the a an this that these those every each some any no',
    'wh_word': 'what who whom whose which where when why how',
    'quantifier': 'all some many much more most few several every each any both',
    'intensifier': 'very really so too quite pretty totally extremely',
}
CLOSED_NAMES = list(CLOSED)
PRONOUNS = set(' '.join(CLOSED[k] for k in ['pron_1sg', 'pron_1pl', 'pron_2', 'pron_3sg', 'pron_3pl']).split())

SUFFIXES = ['ing', 'ed', 's', 'ly', 'tion', 'ment', 'ness', 'er', 'est', 'able', 'ful', 'less',
            'ity', 'ous', 'ive', 'al', 'ize', "n't", "'s", 'y']

# sizes
N_LEX = len(LEXNAMES) + 1  # + no-synset indicator
OFF_SEM = N_LEX
OFF_FW = OFF_SEM + len(SEM_NAMES)
OFF_CL = OFF_FW + len(FUNCTION_WORDS)
OFF_SUF = OFF_CL + len(CLOSED_NAMES)
OFF_LEN = OFF_SUF + len(SUFFIXES)
N_LEN = 7
D_WORD = OFF_LEN + N_LEN

SEM_LOOKUP = {}
for j, k in enumerate(SEM_NAMES):
    for w in SEMANTIC[k].split():
        SEM_LOOKUP.setdefault(w, []).append(j)
CL_LOOKUP = {}
for j, k in enumerate(CLOSED_NAMES):
    for w in CLOSED[k].split():
        CL_LOOKUP.setdefault(w, []).append(j)

_PUNCT = re.compile(r"[^a-z0-9']+")


def clean(tok):
    return _PUNCT.sub('', tok.lower()).strip("'")


def variants(w):
    out = [w]
    for suf, rep in (('ies', 'y'), ('es', ''), ('s', ''), ('ed', ''), ('ed', 'e'), ('ing', ''), ('ing', 'e')):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            out.append(w[:-len(suf)] + rep)
    return out


def syllables(w):
    w = re.sub(r'[^a-z]', '', w)
    if not w:
        return 0
    groups = re.findall(r'[aeiouy]+', w)
    n = len(groups)
    if w.endswith('e') and not w.endswith('le') and n > 1:
        n -= 1
    return max(n, 1)


_cache = {}


def word_vec(w):
    if w in _cache:
        return _cache[w]
    v = np.zeros(D_WORD, dtype=np.float64)
    if w:
        # WordNet lexical-category histogram (normalized over synsets)
        syns = []
        if _wn is not None and w.isalpha():
            try:
                syns = _wn.synsets(w)
            except Exception:
                syns = []
        if syns:
            for s in syns:
                j = LEX_IDX.get(s.lexname())
                if j is not None:
                    v[j] += 1.0 / len(syns)
        else:
            v[len(LEXNAMES)] = 1.0
        # semantic categories
        hit = set()
        for x in variants(w):
            for j in SEM_LOOKUP.get(x, []):
                hit.add(j)
        for j in hit:
            v[OFF_SEM + j] = 1.0
        # function words
        if w in FW_IDX:
            v[OFF_FW + FW_IDX[w]] = 1.0
        # closed class
        for j in CL_LOOKUP.get(w, []):
            v[OFF_CL + j] = 1.0
        if w.endswith("n't"):
            v[OFF_CL + CLOSED_NAMES.index('negation')] = 1.0
        # suffixes
        for j, s in enumerate(SUFFIXES):
            if w.endswith(s) and len(w) > len(s) + 1:
                v[OFF_SUF + j] = 1.0
        # length / syllable cues
        L = len(w)
        sy = syllables(w)
        v[OFF_LEN + 0] = L
        v[OFF_LEN + 1] = sy
        v[OFF_LEN + 2] = np.log1p(L)
        v[OFF_LEN + 3] = float(L >= 7)
        v[OFF_LEN + 4] = float(L <= 3)
        v[OFF_LEN + 5] = float("'" in w)
        v[OFF_LEN + 6] = float(any(c.isdigit() for c in w))
    _cache[w] = v
    return v


def story_arc(N):
    i = np.arange(N, dtype=np.float64)
    p = (i + 0.5) / N
    x = 2 * p - 1
    cols = []
    for k in range(1, 11):
        cols.append(np.sin(2 * np.pi * k * p))
        cols.append(np.cos(2 * np.pi * k * p))
    for k in range(1, 11):
        c = np.zeros(11); c[k] = 1
        cols.append(np.polynomial.chebyshev.chebval(x, c))
    for k in range(1, 11):
        c = np.zeros(11); c[k] = 1
        cols.append(np.polynomial.legendre.legval(x, c))
    for n, w in ((20, 0.5 / 21), (20, 1 / 21), (20, 2 / 21), (5, 0.2 / 6)):
        for j in range(n):
            c = (j + 1) / (n + 1)
            cols.append(np.exp(-0.5 * ((p - c) / w) ** 2))
    cols.append(np.log(i + 1))
    cols.append(np.log(N - i))
    cols.append(p - 0.5)
    A = np.stack(cols, 1)
    sd = A.std(0)
    sd[sd < 1e-12] = 1.0
    A = (A - A.mean(0)) / sd * np.sqrt(0.5)
    return A


TAUS = (5, 20, 100, 500)
DIFFS = ((0, 1), (1, 2), (2, 3))  # short-minus-long pairs over TAUS indices


def content(texts):
    N = len(texts)
    gram_sum = np.zeros((N, D_WORD))
    newest = np.zeros((N, D_WORD))
    extra = []
    seen = set()
    since_sent = 0.0
    since_pron = 0.0
    n_sent = 0
    nov_rate = 0.0
    a_nov = np.exp(-1 / 20.0)
    for t, text in enumerate(texts):
        toks = text.split()
        words = [clean(x) for x in toks]
        for w in words:
            gram_sum[t] += word_vec(w)
        last_tok = toks[-1] if toks else ''
        w = words[-1] if words else ''
        newest[t] = word_vec(w)
        is_new = float(w not in seen)
        seen.add(w)
        nov_rate = a_nov * nov_rate + (1 - a_nov) * is_new
        since_sent += 1
        since_pron += 1
        if w in PRONOUNS:
            since_pron = 0
        extra.append([np.log1p(since_sent), np.log1p(since_pron), is_new, nov_rate,
                      len(seen) / (t + 1.0), float(t), float(len(toks)), float(n_sent)])
        if re.search(r'[.!?]', last_tok):
            since_sent = 0
            n_sent += 1
    extra = np.array(extra)
    # exponentially-decayed story averages of newest-word features
    avgs = []
    dists = np.zeros((N, len(TAUS)))
    for k, tau in enumerate(TAUS):
        a = np.exp(-1.0 / tau)
        m = np.zeros(D_WORD)
        out = np.zeros((N, D_WORD))
        for t in range(N):
            dists[t, k] = np.linalg.norm(newest[t] - m)
            m = a * m + (1 - a) * newest[t]
            out[t] = m
        avgs.append(out)
    diffs = [avgs[s] - avgs[l] for s, l in DIFFS]
    return np.concatenate([gram_sum, newest] + avgs + diffs + [dists, extra], 1)


class Embedder:
    def __init__(self):
        self.mask = None

    def __call__(self, texts):
        texts = list(texts)
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        X = np.concatenate([story_arc(len(texts)), content(texts)], 1)
        if self.mask is None:
            self.mask = X.var(0) > 0.14
        return X[:, self.mask].astype(np.float32)


def build_embedder():
    return Embedder()
