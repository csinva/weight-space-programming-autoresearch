"""Hand-written text features for predicting fMRI responses.

Story-arc block (position features) + content block (WordNet lexnames, 72 semantic categories,
function words, closed-class lists, suffix/length/syllable cues; 10-gram sum, newest word,
exponential story averages, differences, distances, counters) + variance mask from first story.
"""
import re
import numpy as np

MODEL_NAME = 'rec-arc-content-v1'
MODEL_DESCRIPTION = ('Story-arc position block (108 dims, var 0.5) + content block (WordNet lexnames, '
                     '72 semantic cats, function words, closed-class, suffix/length/syllable; 10-gram sum, '
                     'newest word, EMAs tau 5/20/100/500, diffs, distances, counters); var>0.14 mask from first story.')

# ----------------------------------------------------------------------------- word lists
FUNCTION_WORDS = ('the a an and or but of to in on at for with from by as that this it is was be '
                  'have had not i you he she we they my his her so if then there what when just like '
                  'all about up out do would were me').split()

CLOSED = {
    'pronoun': 'i me my mine myself you your yours yourself he him his himself she her hers herself it its itself we us our ours ourselves they them their theirs themselves'.split(),
    'first_sg': 'i me my mine myself'.split(),
    'first_pl': 'we us our ours ourselves'.split(),
    'second': 'you your yours yourself yourselves'.split(),
    'third': 'he him his himself she her hers herself they them their theirs themselves it its'.split(),
    'modal': 'can could may might must shall should will would ought'.split(),
    'negation': "not no never nothing nobody none neither nor nowhere don't didn't doesn't can't won't wouldn't couldn't shouldn't isn't wasn't aren't weren't haven't hasn't hadn't cannot".split(),
    'mental': 'think thought know knew known believe believed feel felt remember remembered forget forgot realize realized wonder wondered understand understood guess guessed imagine imagined decide decided want wanted hope hoped wish wished mean meant suppose figure figured notice noticed'.split(),
    'deixis': 'here there now then this that these those today tomorrow yesterday tonight'.split(),
    'discourse': 'well so okay ok oh um uh like anyway actually basically really right yeah yes alright because but however although'.split(),
    'wh': 'what who whom whose which where when why how'.split(),
    'prep': 'in on at to from with by for of about into onto over under through across after before between behind around against during without within upon toward towards'.split(),
    'conj': 'and or but nor yet so because although though while whereas unless since if'.split(),
    'det': 'the a an this that these those some any every each no another'.split(),
    'aux': 'is am are was were be been being have has had do does did'.split(),
    'quant': 'all some many much few more most less least several lots lot both enough every'.split(),
}

SEM = {
    'family': 'mother mom mum father dad parent parents son daughter brother sister sibling family wife husband uncle aunt cousin grandmother grandfather grandma grandpa baby child children kid kids'.split(),
    'friends_social': 'friend friends buddy guy guys people person neighbor stranger crowd group team partner boyfriend girlfriend roommate colleague everyone somebody someone'.split(),
    'body': 'body head face eye eyes hand hands arm arms leg legs foot feet hair mouth nose ear ears finger fingers skin heart back neck shoulder knee chest stomach teeth tooth lip lips blood bone'.split(),
    'health': 'sick ill hospital doctor nurse pain hurt disease cancer medicine pill surgery injury healthy health died die death dying dead wound fever'.split(),
    'emotion_pos': 'happy happiness joy glad love loved loving excited proud fun laugh laughed smile smiled pleased delighted wonderful great amazing beautiful lovely enjoy enjoyed'.split(),
    'emotion_neg': 'sad angry afraid scared fear upset hate hated cry cried worried anxious nervous lonely miserable terrible awful horrible depressed ashamed embarrassed guilty frustrated'.split(),
    'anger_conflict': 'fight fought argue argued yell yelled scream screamed hit kill killed attack war enemy violence gun shot punch angry mad furious'.split(),
    'perception_see': 'see saw seen look looked looking watch watched notice noticed stare stared glance view sight visible appear appeared'.split(),
    'perception_hear': 'hear heard listen listened sound sounds noise loud quiet silent silence voice music song sing sang'.split(),
    'perception_touch': 'touch touched feel felt hold held grab grabbed soft hard rough smooth warm cold hot wet dry'.split(),
    'speech': 'say said says tell told talk talked talking speak spoke ask asked answer answered call called reply replied shout whisper explain explained conversation word words'.split(),
    'cognition': 'think thought know knew idea mind reason understand understood learn learned question decide decided plan believe memory remember realize'.split(),
    'motion': 'go went gone going come came coming walk walked run ran move moved drive drove fly flew leave left arrive arrived travel traveled climb climbed jump jumped fall fell follow followed'.split(),
    'posture': 'sit sat stand stood lie lay lying sleep slept wake woke rest kneel lean leaned'.split(),
    'place_home': 'house home room kitchen bedroom bathroom door window wall floor bed couch table apartment yard garage roof stairs'.split(),
    'place_city': 'city town street road building store shop restaurant bar office school church hotel park downtown block neighborhood village'.split(),
    'place_nature': 'tree trees forest woods river lake ocean sea beach mountain hill field grass desert island sky sun moon star stars rain snow wind'.split(),
    'geography': 'america american country state california new york london europe world china africa mexico texas france england india canada'.split(),
    'vehicle': 'car cars truck bus train plane airplane boat ship bike bicycle taxi cab subway airport station'.split(),
    'time_unit': 'time day days night week weeks month months year years hour hours minute minutes second seconds moment morning evening afternoon weekend'.split(),
    'time_rel': 'now then later soon before after ago already still yet again always never sometimes often once until while during eventually suddenly finally'.split(),
    'number': 'one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty hundred thousand million first second third half dozen number'.split(),
    'money': 'money dollar dollars pay paid cost buy bought sell sold price cash bank rich poor job work worked salary rent debt'.split(),
    'work': 'work job boss office company business career employee manager project meeting worked working hired fired'.split(),
    'school': 'school college university class teacher student students professor study studied homework grade test exam degree learn'.split(),
    'food': 'food eat ate eating dinner lunch breakfast meal cook cooked kitchen bread meat chicken coffee tea pizza cake drink drank water wine beer milk'.split(),
    'clothing': 'shirt dress shoes shoe coat jacket hat pants jeans clothes wear wore wearing suit uniform'.split(),
    'animal': 'dog dogs cat cats horse bird birds fish animal animals cow pig bear snake mouse rat chicken wolf lion'.split(),
    'color': 'red blue green yellow black white brown gray grey pink purple orange color colors dark bright'.split(),
    'size': 'big small large little huge tiny tall short long wide narrow giant enormous'.split(),
    'quantity_deg': 'very really so too quite pretty extremely totally completely just almost barely much more most less'.split(),
    'religion': 'god church pray prayed prayer jesus christ heaven hell soul spirit faith religious holy bible priest'.split(),
    'law_crime': 'police cop cops arrest arrested jail prison crime criminal court judge lawyer law steal stole thief guilty trial'.split(),
    'politics': 'government president politics political vote election country nation law power congress war military army soldier'.split(),
    'technology': 'phone computer internet email online video camera tv television radio machine screen website text texted app'.split(),
    'media_art': 'book books story stories read wrote write writing film movie show picture photo paint art artist music band song'.split(),
    'game_sport': 'game games play played playing sport ball team win won lose lost score football baseball basketball soccer'.split(),
    'sex_romance': 'kiss kissed love lover date dating sex married marry marriage wedding romantic boyfriend girlfriend relationship'.split(),
    'gender_male': 'he him his man men boy boys guy father brother husband son mr king'.split(),
    'gender_female': 'she her hers woman women girl girls lady mother sister wife daughter mrs queen'.split(),
    'age': 'old young age older younger baby child teenager adult kid elderly born birthday'.split(),
    'certainty': 'sure certain definitely absolutely clearly obviously exactly true truth fact really'.split(),
    'tentative': 'maybe perhaps probably possibly might seem seemed seems guess somehow sort kind almost apparently'.split(),
    'causation': 'because cause caused reason why since therefore so effect result made make makes'.split(),
    'achievement': 'win won success successful achieve achieved goal best try tried effort accomplish failed fail failure'.split(),
    'power': 'power control boss leader lead led command order ordered strong weak force forced rule'.split(),
    'risk_danger': 'danger dangerous risk safe safety threat afraid careful accident emergency crash fire'.split(),
    'death': 'die died dead death kill killed funeral grave buried murder suicide corpse'.split(),
    'home_objects': 'bed chair table door key keys box bag bottle cup glass plate lamp book paper letter'.split(),
    'tools_objects': 'knife gun rope tool hammer machine wheel wire pen pencil card bag'.split(),
    'weather': 'rain snow storm wind sun sunny cloud clouds cold hot weather winter summer spring fall'.split(),
    'light_dark': 'light dark darkness bright shadow shine shining lamp candle glow'.split(),
    'space_dir': 'up down left right inside outside front back top bottom side middle above below near far behind'.split(),
    'communication_media': 'letter call called phone message note email text mail news newspaper'.split(),
    'question_words': 'what why how who where when which whether'.split(),
    'affirmation': 'yes yeah okay ok sure right alright fine agree'.split(),
    'swear_informal': 'damn hell shit fuck crap stupid crazy dumb'.split(),
    'filler': 'um uh like oh well you know mean'.split(),
    'possession': 'have has had own owned get got give gave take took keep kept'.split(),
    'change': 'change changed become became turn turned grow grew start started begin began stop stopped end ended'.split(),
    'help_care': 'help helped care cared support protect save saved need needed'.split(),
    'desire': 'want wanted wish hope hoped need desire dream dreamed'.split(),
    'trust_honesty': 'trust honest lie lied truth secret promise promised believe'.split(),
    'surprise': 'surprise surprised suddenly shock shocked amazed weird strange wow'.split(),
    'fear_anx': 'afraid fear scared terrified panic nervous worry worried anxious'.split(),
    'achieve_effort': 'try tried work hard effort practice practiced train trained'.split(),
    'hosp_med': 'hospital doctor nurse patient surgery medicine drug drugs pill'.split(),
    'water': 'water river lake ocean sea swim swam wave waves boat beach wet'.split(),
    'travel': 'trip travel traveled journey flight vacation visit visited tour abroad'.split(),
    'party_social': 'party dance danced drink drinking bar club friends celebrate'.split(),
    'hair_appearance': 'hair pretty handsome ugly beautiful wear wore look looks dressed'.split(),
    'abstract': 'life world thing things way idea sense truth meaning reality nature fact'.split(),
}
assert len(SEM) == 72, len(SEM)

SUFFIXES = ['ing', 'ed', 'ly', 's', 'es', 'tion', 'sion', 'ness', 'ment', 'er', 'est', 'able', 'ible', 'ful',
            'less', 'ous', 'ive', 'al', 'ity', 'ize', 'ism', 'ist', "n't", "'s", 'y']

PRONOUNS = set(CLOSED['pronoun'])

try:
    from nltk.corpus import wordnet as wn
    LEXNAMES = sorted(set(s.lexname() for s in wn.all_synsets()))
except Exception:  # pragma: no cover
    wn = None
    LEXNAMES = []


def norm(tok):
    return re.sub(r"[^a-z']", '', tok.lower()).strip("'")


def stems(w):
    out = [w]
    for suf, rep in (('ies', 'y'), ('es', ''), ('s', ''), ('ed', ''), ('ed', 'e'), ('ing', ''), ('ing', 'e'), ('ly', '')):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            out.append(w[:-len(suf)] + rep)
    return out


def syllables(w):
    w = re.sub(r"[^a-z]", '', w)
    if not w:
        return 0
    n = len(re.findall(r'[aeiouy]+', w))
    if w.endswith('e') and not w.endswith('le') and n > 1:
        n -= 1
    return max(n, 1)


# feature index layout
LEX_IDX = {n: i for i, n in enumerate(LEXNAMES)}
OFF_LEX = 0
OFF_SEM = OFF_LEX + len(LEXNAMES)
OFF_FW = OFF_SEM + len(SEM)
OFF_CL = OFF_FW + len(FUNCTION_WORDS)
OFF_SUF = OFF_CL + len(CLOSED)
OFF_LEN = OFF_SUF + len(SUFFIXES)
N_LEN = 6
D = OFF_LEN + N_LEN

SEM_SETS = [set(v) for v in SEM.values()]
CLOSED_SETS = [set(v) for v in CLOSED.values()]
FW_IDX = {w: i for i, w in enumerate(FUNCTION_WORDS)}

_cache = {}


def word_features(w):
    if w in _cache:
        return _cache[w]
    f = np.zeros(D, dtype=np.float64)
    if w:
        # WordNet lexical-category histogram (normalized over synsets)
        if wn is not None and w not in FW_IDX:
            try:
                syns = wn.synsets(w.replace("'", ''))
            except Exception:
                syns = []
            if syns:
                for s in syns:
                    f[OFF_LEX + LEX_IDX[s.lexname()]] += 1.0 / len(syns)
        st = stems(w)
        for j, S in enumerate(SEM_SETS):
            if any(x in S for x in st):
                f[OFF_SEM + j] = 1
        if w in FW_IDX:
            f[OFF_FW + FW_IDX[w]] = 1
        for j, S in enumerate(CLOSED_SETS):
            if w in S:
                f[OFF_CL + j] = 1
        for j, s in enumerate(SUFFIXES):
            if w.endswith(s) and len(w) > len(s) + 1:
                f[OFF_SUF + j] = 1
        L = len(w); sy = syllables(w)
        f[OFF_LEN:OFF_LEN + N_LEN] = [L, np.log1p(L), float(L >= 7), sy, float(sy >= 3), float(L <= 3)]
    _cache[w] = f
    return f


# ----------------------------------------------------------------------------- story arc
def arc_block(N):
    i = np.arange(N, dtype=np.float64)
    p = (i + 0.5) / N
    x = 2 * p - 1
    cols = []
    for k in range(1, 11):
        cols += [np.sin(2 * np.pi * k * p), np.cos(2 * np.pi * k * p)]
    cols += [np.polynomial.chebyshev.chebval(x, [0] * k + [1]) for k in range(1, 11)]
    cols += [np.polynomial.legendre.legval(x, [0] * k + [1]) for k in range(1, 11)]
    for n, w in ((20, 0.5 / 21), (20, 1 / 21), (20, 2 / 21), (5, 0.2 / 6)):
        for c in (np.arange(n) + 1) / (n + 1):
            cols.append(np.exp(-0.5 * ((p - c) / w) ** 2))
    cols += [np.log(i + 1), np.log(N - i), p - 0.5]
    A = np.stack(cols, 1)
    sd = A.std(0)
    sd[sd < 1e-12] = 1
    A = (A - A.mean(0)) / sd * np.sqrt(0.5)
    return A


# ----------------------------------------------------------------------------- embedder
class Embedder:
    TAUS = (5, 20, 100, 500)

    def __init__(self):
        self.mask = None

    def content(self, texts):
        N = len(texts)
        toks = [t.split() for t in texts]
        last_raw = [t[-1] if t else '' for t in toks]
        last = [norm(w) for w in last_raw]
        F = np.stack([word_features(w) for w in last])  # newest word, N x D
        # 10-gram sum via cumulative sums over story words
        C = np.vstack([np.zeros((1, D)), np.cumsum(F, 0)])
        k = np.array([min(len(t), i + 1) for i, t in enumerate(toks)])
        idx = np.arange(N)
        G = C[idx + 1] - C[idx + 1 - k]
        # exponential story averages (normalized), distance of new word from previous average
        emas, dists = [], []
        for tau in self.TAUS:
            a = 1 - np.exp(-1.0 / tau)
            m = np.zeros(D); wsum = 0.0
            M = np.zeros((N, D)); dist = np.zeros(N)
            for t in range(N):
                prev = m / wsum if wsum > 0 else np.zeros(D)
                dist[t] = np.linalg.norm(F[t] - prev)
                m = (1 - a) * m + a * F[t]; wsum = (1 - a) * wsum + a
                M[t] = m / wsum
            emas.append(M); dists.append(dist)
        diffs = [emas[0] - emas[2], emas[1] - emas[3]]
        # counters
        since_sent = np.zeros(N); since_pron = np.zeros(N)
        is_new = np.zeros(N); n_types = np.zeros(N)
        ls = lp = -1; seen = set()
        for t in range(N):
            if last[t] in PRONOUNS:
                lp = t
            since_sent[t] = np.log1p(t - ls if ls >= 0 else t + 1)
            since_pron[t] = np.log1p(t - lp if lp >= 0 else t + 1)
            if re.search(r'[.!?]["\']?$', last_raw[t]):
                ls = t
            if last[t] and last[t] not in seen:
                is_new[t] = 1; seen.add(last[t])
            n_types[t] = len(seen)
        a = 1 - np.exp(-1.0 / 20)
        nov = np.zeros(N); m = 0.0; ws = 0.0
        for t in range(N):
            m = (1 - a) * m + a * is_new[t]; ws = (1 - a) * ws + a
            nov[t] = m / ws
        pos = np.arange(N, dtype=np.float64)
        extra = np.stack(dists + [since_sent, since_pron, is_new, nov, n_types / (pos + 1),
                                  pos + 1, n_types, np.array([len(t) for t in toks], float)], 1)
        return np.hstack([G, F] + emas + diffs + [extra])

    def __call__(self, texts):
        N = len(texts)
        X = np.hstack([arc_block(N), self.content(texts)])
        if self.mask is None:  # variance mask fixed from first story
            self.mask = X.var(0) > 0.14
        return X[:, self.mask].astype(np.float32)


def build_embedder():
    return Embedder()
