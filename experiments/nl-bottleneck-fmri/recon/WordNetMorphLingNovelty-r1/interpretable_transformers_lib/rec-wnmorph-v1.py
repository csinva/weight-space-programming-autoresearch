"""Hand-written text features for predicting fMRI responses (no training).

Reconstruction of a position-dominated handcrafted model: story-position bases, sentence and
"paragraph" structure, punctuation decays, WordNet lexname content, semantic category lists,
closed-class word lists, suffix rules, letter phonology, running averages and novelty features.
"""
import re
import numpy as np

MODEL_NAME = 'rec-wnmorph-v1'
MODEL_DESCRIPTION = ('Position (Fourier/Chebyshev/Legendre/Gaussian bumps/log edges, var 0.5), sentence and '
                     'merged-paragraph units, punctuation decays tau 2-200, WordNet 45 lexnames direct+expanded, '
                     '72 hand semantic categories, word lists, suffixes, letter phonology summed over 10-gram and '
                     'EMAs tau 5-500, novelty; low-variance dims dropped on first story.')

try:
    from nltk.corpus import wordnet as wn
    wn.synsets('dog')
except Exception:  # pragma: no cover
    wn = None

LEXNAMES = ['adj.all', 'adj.pert', 'adv.all', 'noun.Tops', 'noun.act', 'noun.animal', 'noun.artifact',
            'noun.attribute', 'noun.body', 'noun.cognition', 'noun.communication', 'noun.event', 'noun.feeling',
            'noun.food', 'noun.group', 'noun.location', 'noun.motive', 'noun.object', 'noun.person',
            'noun.phenomenon', 'noun.plant', 'noun.possession', 'noun.process', 'noun.quantity', 'noun.relation',
            'noun.shape', 'noun.state', 'noun.substance', 'noun.time', 'verb.body', 'verb.change',
            'verb.cognition', 'verb.communication', 'verb.competition', 'verb.consumption', 'verb.contact',
            'verb.creation', 'verb.emotion', 'verb.motion', 'verb.perception', 'verb.possession', 'verb.social',
            'verb.stative', 'verb.weather', 'adj.ppl']
LEX_IDX = {n: i for i, n in enumerate(LEXNAMES)}

_CATS_RAW = """
body: body head face eye eyes hand hands arm arms leg legs foot feet finger fingers hair skin mouth nose ear ears neck back chest heart blood bone shoulder knee lip lips tooth teeth stomach belly
family: mother father mom dad mommy daddy parent parents brother sister son daughter wife husband family grandmother grandfather grandma grandpa aunt uncle cousin baby kids kid child children
friend: friend friends buddy pal partner roommate neighbor neighbors guy guys people person someone everybody everyone
romance: love lover kiss kissed date dating boyfriend girlfriend married marry wedding romance romantic sex sexy crush heart
male: he him his man men boy boys guy gentleman sir mr king
female: she her hers woman women girl girls lady ladies mrs ms miss queen
posemo: happy joy glad love loved great good wonderful amazing beautiful nice fun laugh smile excited proud hope lucky fantastic
negemo: sad bad terrible awful horrible upset hurt pain cry cried crying tears lonely miserable depressed sorry worse worst
fear: afraid fear scared scary terrified panic nervous anxious worried worry frightened danger dangerous horror
anger: angry mad furious rage hate hated yell yelled scream screamed fight annoyed pissed
humor: funny joke jokes laugh laughed laughing hilarious comedy silly ridiculous
number: one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty hundred thousand million first second third
time: time day days night week weeks month months year years hour hours minute minutes moment later soon now then before after ago today tomorrow yesterday
timeofday: morning afternoon evening night midnight noon dawn dusk tonight sunrise sunset
calendar: monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november december weekend
season: summer winter spring fall autumn christmas holiday holidays birthday
place: place room house home building city town street road country world school office church hospital store kitchen bedroom bathroom door window
city: city town street streets downtown york london chicago paris neighborhood block apartment subway bus traffic
nature: tree trees forest woods mountain mountains river lake ocean sea beach sky sun moon star stars field grass flower flowers island desert
water: water river lake ocean sea rain swim swimming wave waves boat ship wet pool drown
weather: rain snow wind storm cold hot weather cloud clouds sunny thunder ice freezing
animal: dog dogs cat cats horse horses bird birds fish cow pig animal animals bear wolf snake mouse rat lion chicken
food: food eat ate eating dinner lunch breakfast meal bread meat cheese pizza coffee tea cake chocolate apple restaurant cook cooking hungry
drink: drink drank drinking beer wine whiskey vodka bar drunk alcohol bottle glass
clothing: shirt pants dress shoes shoe hat coat jacket clothes jeans suit tie wear wearing wore socks
color: red blue green yellow black white brown pink purple orange gray grey color colors dark bright
vehicle: car cars truck bus train plane airplane bike bicycle drive drove driving ride flight airport taxi
motion: go went going come came walk walked walking run ran running move moved jump fell fall climb travel
speech: say said says tell told talk talked talking speak spoke ask asked answer call called shout whisper
cognition: think thought know knew believe remember forgot forget understand idea mind realize realized wonder imagine guess decide
vision: see saw seen look looked looking watch watched watching eye eyes notice stare stared glance view
sound: hear heard listen listened sound sounds noise loud quiet silence voice music song sing sang
touch: touch touched feel felt hold held grab grabbed push pushed pull pulled hit soft hard
smelltaste: smell smelled taste tasted sweet sour bitter delicious stink
money: money dollar dollars pay paid buy bought sell sold cost price cash rich poor bank job
work: work worked working job boss office business company career meeting hired fired employee
school: school class teacher student students college university study homework test grade professor lesson
writing: write wrote written book books letter letters read reading page story stories paper word words
music: music song songs band play played guitar piano sing singer concert dance dancing
sport: game games team play ball football baseball basketball soccer win won lose lost score coach
art: art artist paint painting picture photo photograph camera film movie movies show theater
science: science scientist experiment research data lab theory study physics brain
technology: computer phone internet email video tv television radio machine robot screen
medicine: doctor doctors hospital nurse sick ill disease cancer medicine surgery patient pill pills health
death: death die died dead dying kill killed funeral grave murder suicide body
violence: gun guns shot shoot shooting knife blood fight attack beat beaten hit violence weapon
war: war army soldier soldiers military battle enemy bomb troops navy
law: law police cop cops arrest arrested prison jail judge court lawyer crime criminal guilty trial
politics: government president election vote politics political state nation country power
religion: god church pray prayer heaven hell religion religious jesus christian jewish faith soul spirit
sleep: sleep slept sleeping bed dream dreamed dreams tired wake woke awake nap
age: old young age older younger new teenager adult elderly years
size: big small large little huge tiny tall short long giant
quantity: all some many much more most few less lot lots every each several whole half
space: up down in out over under above below inside outside top bottom front behind near far left right
direction: north south east west forward backward around across through toward away
shape: round square circle line straight flat curve edge corner
temperature: hot cold warm cool heat freezing burning fire
light: light dark bright shine shining lamp candle shadow shadows glow
home: home house apartment room bed kitchen couch table chair door wall floor
tool: tool tools knife hammer key keys box bag rope machine
game: game games play played card cards toy toys puzzle win
greeting: hello hi hey bye goodbye thanks thank please welcome yes
swear: damn hell shit fuck fucking crap ass god
assent: yes yeah okay ok sure right exactly absolutely alright
filler: um uh like well so oh ah hmm anyway actually basically
certainty: always never certainly definitely sure surely really absolutely completely totally
tentative: maybe perhaps possibly probably might seem seemed guess sort kind somehow
cause: because why reason cause caused since so therefore effect result
contrast: but however although though yet instead despite whereas
question: what who where when why how which whose whom
achievement: win won success succeed achieve goal effort try tried trying best champion
power: power control strong weak boss leader rule command force
social: people friend friends party together group meet met everyone community crowd
help: help helped helping save saved support care protect rescue
conflict: argue argument fight fought problem trouble against enemy blame
change: change changed become became turn turned grow grew begin began start started end ended
possession: have had has get got give gave take took keep kept lose lost own
perception: feel felt seem seemed appear looked sounded noticed sense
desire: want wanted need needed wish wished hope hoped desire like liked
emotionverb: love loved hate hated enjoy enjoyed miss missed fear care
mental_state: confused curious surprised shocked amazed embarrassed ashamed guilty
travel: trip travel traveled journey vacation flight hotel abroad visit visited tour
building: building house tower bridge wall roof floor stairs hall office
material: wood metal stone glass paper plastic gold iron steel
"""
CATS = []
CAT_WORDS = {}
for _line in _CATS_RAW.strip().splitlines():
    _n, _ws = _line.split(':', 1)
    CATS.append(_n.strip())
    for _w in _ws.split():
        CAT_WORDS.setdefault(_w, set()).add(len(CATS) - 1)
N_CATS = len(CATS)

WORDLISTS = {
    'pron1s': 'i me my mine myself',
    'pron1p': 'we us our ours ourselves',
    'pron2': 'you your yours yourself yourselves',
    'pron3s': 'he him his himself she her hers herself it its itself',
    'pron3p': 'they them their theirs themselves',
    'demonstr': 'this that these those here there',
    'modal': 'can could may might must shall should will would ought',
    'negation': "not no never nothing nobody none nowhere neither nor n't don't didn't can't won't wasn't isn't couldn't wouldn't shouldn't doesn't",
    'aux': 'be am is are was were been being have has had do does did',
    'article': 'a an the',
    'prep': 'in on at by for with from to of about into over under after before through between',
    'conj': 'and or but so because if when while although though',
    'wh': 'what who where when why how which',
    'quant': 'all some any many much more most few each every',
}
WL_NAMES = list(WORDLISTS)
WL_SETS = [set(WORDLISTS[k].split()) for k in WL_NAMES]

SUFFIXES = ['ing', 'ed', 'ly', 's', 'es', 'er', 'est', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful',
            'less', 'ous', 'ive', 'al', 'ity', 'ize', 'ish', "'s", "n't"]
VOWELS = set('aeiouy')


def clean(w):
    w = w.lower()
    w = re.sub(r"[^a-z0-9']", '', w)
    return w.strip("'")


def lemma_guess(w):
    for suf, rep in (('ies', 'y'), ('ing', ''), ('ed', ''), ('es', ''), ('s', '')):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)] + rep
    return w


def n_syllables(w):
    groups = re.findall(r'[aeiouy]+', w)
    n = len(groups)
    if w.endswith('e') and n > 1 and not w.endswith('le'):
        n -= 1
    return max(n, 1 if w else 0)


class Embedder:
    def __init__(self):
        self.cache = {}
        self.keep = None

    # ---------------- word-level content features ----------------
    def word_feats(self, w):
        if w in self.cache:
            return self.cache[w]
        direct = np.zeros(len(LEXNAMES))
        expanded = np.zeros(len(LEXNAMES))
        if wn is not None and w and w.isalpha():
            try:
                syns = wn.synsets(w)
            except Exception:
                syns = []
            if syns:
                for s in syns:
                    direct[LEX_IDX.get(s.lexname(), 0)] += 1.0 / len(syns)
                expanded += direct
                for s in syns[:3]:
                    hyps = s.hypernyms() + s.instance_hypernyms()
                    for h in hyps:
                        expanded[LEX_IDX.get(h.lexname(), 0)] += 0.5 / (len(hyps) * min(3, len(syns)))
                    gloss = [g for g in re.findall(r'[a-z]+', s.definition().lower()) if len(g) > 3]
                    gl = 0
                    gvec = np.zeros(len(LEXNAMES))
                    for g in gloss:
                        gs = wn.synsets(g)
                        if gs:
                            gvec[LEX_IDX.get(gs[0].lexname(), 0)] += 1
                            gl += 1
                    if gl:
                        expanded += 0.5 * gvec / gl / min(3, len(syns))
        cats = np.zeros(N_CATS)
        for key in {w, lemma_guess(w)}:
            for c in CAT_WORDS.get(key, ()):
                cats[c] = 1
        wl = np.array([1.0 if w in s else 0.0 for s in WL_SETS])
        suf = np.array([1.0 if (w.endswith(s) and len(w) > len(s) + 2) else 0.0 for s in SUFFIXES])
        letters = [c for c in w if c.isalpha()]
        nv = sum(c in VOWELS for c in letters)
        clusters = re.findall(r'[^aeiouy\W\d]{2,}', w)
        phon = np.array([
            len(letters), n_syllables(w), nv, len(letters) - nv,
            1.0 if letters and letters[0] in VOWELS else 0.0,
            1.0 if letters and letters[-1] in VOWELS else 0.0,
            len(clusters), max([len(c) for c in clusters], default=0),
            1.0 if w else 0.0,
        ])
        f = np.concatenate([direct, expanded, cats, wl, suf, phon])
        self.cache[w] = f
        return f

    # ---------------- position helpers ----------------
    @staticmethod
    def scale(m):
        m = np.asarray(m, dtype=np.float64)
        if m.ndim == 1:
            m = m[:, None]
        sd = m.std(0)
        sd[sd < 1e-12] = 1.0
        return (m - m.mean(0)) / sd * np.sqrt(0.5)

    def position_feats(self, n):
        idx = np.arange(n, dtype=np.float64)
        p = idx / max(n, 1)
        x = 2 * p - 1
        cols = []
        for k in range(1, 11):
            cols += [np.sin(2 * np.pi * k * p), np.cos(2 * np.pi * k * p)]
        cheb = np.polynomial.chebyshev.chebvander(x, 10)[:, 1:]
        leg = np.polynomial.legendre.legvander(x, 10)[:, 1:]
        bumps = []
        for width, nc in ((0.03, 33), (0.06, 17), (0.12, 10), (0.25, 5)):
            for c in np.linspace(0, 1, nc):
                bumps.append(np.exp(-0.5 * ((p - c) / width) ** 2))
        cols += list(cheb.T) + list(leg.T) + bumps
        cols += [np.log1p(idx), np.log1p(n - 1 - idx)]
        return self.scale(np.stack(cols, 1))

    @staticmethod
    def unit_feats(bounds, n):
        """bounds: list of (start, end_exclusive) covering 0..n."""
        out = np.zeros((n, 9))
        for s, e in bounds:
            L = e - s
            j = np.arange(L, dtype=np.float64)
            r = j / max(L - 1, 1)
            out[s:e, 0] = r
            out[s:e, 1] = np.log1p(j)
            out[s:e, 2] = np.log1p(L - 1 - j)
            out[s:e, 3] = np.log(L)
            out[s, 4] = 1
            out[e - 1, 5] = 1
            out[s:e, 6] = np.sin(np.pi * r)
            out[s:e, 7] = np.cos(np.pi * r)
            out[s:e, 8] = np.sin(2 * np.pi * r)
        return out

    def structure_feats(self, raw):
        n = len(raw)
        ends = {'.': [], '?': [], '!': []}
        is_end = np.zeros(n, bool)
        for i, t in enumerate(raw):
            t2 = t.rstrip('"\')]')
            if t2 and t2[-1] in '.?!':
                is_end[i] = True
                ends[t2[-1]].append(i)
        # sentences
        sents, s = [], 0
        for i in range(n):
            if is_end[i]:
                sents.append((s, i + 1)); s = i + 1
        if s < n:
            sents.append((s, n))
        blocks = [self.unit_feats(sents, n)]
        for minlen in (10, 20, 40, 80, 160, 320):
            units, cs, cl = [], None, 0
            for a, b in sents:
                if cs is None:
                    cs = a
                cl += b - a
                if cl >= minlen:
                    units.append((cs, b)); cs, cl = None, 0
            if cs is not None:
                if units:
                    units[-1] = (units[-1][0], n)
                else:
                    units.append((cs, n))
            blocks.append(self.unit_feats(units, n))
        # decays
        taus = (2, 5, 10, 20, 50, 100, 200)
        def decays(pos):
            pos = np.array(sorted(pos), dtype=np.int64)
            res = np.zeros((n, 2 * len(taus)))
            if len(pos) == 0:
                return res
            idx = np.arange(n)
            k = np.searchsorted(pos, idx, side='right') - 1
            since = np.where(k >= 0, idx - pos[np.clip(k, 0, None)], np.inf)
            k2 = np.searchsorted(pos, idx, side='left')
            until = np.where(k2 < len(pos), pos[np.clip(k2, None, len(pos) - 1)] - idx, np.inf)
            for t_i, tau in enumerate(taus):
                res[:, t_i] = np.exp(-since / tau)
                res[:, len(taus) + t_i] = np.exp(-until / tau)
            return res
        blocks.append(decays(np.where(is_end)[0]))
        for ch in '.?!':
            blocks.append(decays(ends[ch]))
        blocks.append(np.stack([is_end] + [np.isin(np.arange(n), ends[c]) for c in '.?!'], 1).astype(float))
        return np.concatenate(blocks, 1)

    # ---------------- main ----------------
    def __call__(self, texts):
        n = len(texts)
        grams = [t.split() for t in texts]
        raw = [g[-1] if g else '' for g in grams]
        words = [clean(t) for t in raw]
        W = np.stack([self.word_feats(w) for w in words]) if n else np.zeros((0, 1))
        D = W.shape[1]

        # sum over the 10-gram
        S = np.zeros_like(W)
        for i, g in enumerate(grams):
            for t in g:
                S[i] += self.word_feats(clean(t))
        # running averages
        emas = []
        for tau in (5, 20, 100, 500):
            a = 1.0 / tau
            E = np.zeros_like(W)
            acc = np.zeros(D)
            for i in range(n):
                acc = (1 - a) * acc + a * W[i]
                E[i] = acc
            emas.append(E)

        # novelty
        last = {}
        types = set()
        nov = np.zeros((n, 7))
        for i, w in enumerate(words):
            first = w not in last
            nov[i, 0] = 1.0 if first else 0.0
            nov[i, 1] = np.log1p(i - last[w]) if not first else np.log1p(i + 1)
            last[w] = i
            types.add(w)
            nov[i, 2] = len(types) / (i + 1)              # cumulative TTR
            g = [clean(t) for t in grams[i]]
            nov[i, 3] = len(set(g)) / max(len(g), 1)        # 10-gram TTR
            lo = max(0, i - 49)
            nov[i, 4] = len(set(words[lo:i + 1])) / (i + 1 - lo)  # 50-word TTR
            nov[i, 5] = len(types)                          # raw type count (leaks length)
            nov[i, 6] = len(g)                              # raw 10-gram length
        nov_ema = []
        for tau in (5, 20, 100, 500):
            a = 1.0 / tau
            acc = 0.0
            col = np.zeros(n)
            for i in range(n):
                acc = (1 - a) * acc + a * nov[i, 0]
                col[i] = acc
            nov_ema.append(col)

        X = np.concatenate([self.position_feats(n), self.structure_feats(raw), S] + emas +
                           [nov, np.stack(nov_ema, 1)], 1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.keep is None:
            self.keep = X.var(0) >= 0.05
        return X[:, self.keep].astype(np.float32)


def build_embedder():
    return Embedder()
