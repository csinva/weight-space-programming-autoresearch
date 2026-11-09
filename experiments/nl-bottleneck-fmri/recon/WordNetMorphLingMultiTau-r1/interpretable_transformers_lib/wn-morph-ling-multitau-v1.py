"""Hand-written text features for predicting fMRI responses (untrained, lexical + WordNet + multi-timescale).

build_embedder() returns an object mapping a list of 10-gram strings (one story, in order) to an
array (n_10grams, n_features).
"""
import math
import re
import numpy as np

MODEL_NAME = 'wn-morph-ling-multitau-v1'
MODEL_DESCRIPTION = ('Untrained lexical features: WordNet lexname histograms (plain + extended with synonyms/'
                     'hypernyms/gloss words), rarity/depth, 72 semantic categories, 50 function words, 14 affixes, '
                     '19 closed-class sets, phonology, mental/deixis/discourse lists; near-uniform 10-gram sum + last '
                     'word + tense/punct/entropy; EMAs (tau 5,20,100,500) of newest word, 5-500 diff, distance/cosine '
                     'surprise, log steps since sentence end/pronoun, novelty, position; drop var<0.0022 on story 1.')

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

_CATS = """
family: mother father mom dad mum brother sister son daughter wife husband aunt uncle cousin grandmother grandfather grandma grandpa parent parents family baby kids
person: man woman boy girl guy people person child kid lady gentleman friend stranger neighbor someone everyone somebody adult teenager
body: hand hands face eyes eye head arm arms leg legs foot feet hair mouth heart back body finger fingers skin shoulder neck chest stomach knee teeth nose ear ears lips
joy: happy glad joy laugh laughed laughing smile smiled fun love loved delighted excited cheerful wonderful great amazing proud pleased
sad: sad cry cried crying tears grief lonely miserable depressed sorrow upset hurt pain unhappy broken heartbroken mourn
fear: afraid scared fear terrified nervous anxious worried panic frightened dread horror worry scary terror
anger: angry mad furious rage hate hated annoyed yell yelled scream screamed shout shouted fight argue argued
love: love loved kiss kissed hug hugged romantic date dating boyfriend girlfriend married marriage wedding sex lover crush
food: food eat ate eating dinner lunch breakfast bread meat cheese pizza cake apple chicken rice soup sandwich meal cook cooked kitchen restaurant
drink: drink drank water coffee tea beer wine milk juice bottle glass bar alcohol drunk vodka whiskey cup
animal: dog dogs cat cats horse bird birds fish cow pig snake bear wolf mouse rat animal animals lion monkey chicken deer bug
plant: tree trees flower flowers grass plant plants leaf leaves garden forest wood woods seed bush rose
nature: sky sun moon star stars mountain river lake ocean sea beach island hill field rock rocks earth desert valley land
weather: rain snow wind storm cloud clouds weather cold hot warm thunder lightning fog ice sunny freezing
color: red blue green yellow black white brown pink purple orange gray grey color colors dark bright golden
number: one two three four five six seven eight nine ten hundred thousand million twenty thirty fifty first second third
time_unit: minute minutes hour hours day days week weeks month months year years second seconds moment decade century
day_part: morning night evening afternoon tonight today yesterday tomorrow midnight noon dawn dusk sunset weekend
building: house home building school church hospital office store shop hotel apartment room hall library prison station bank
room: room kitchen bedroom bathroom door window wall floor ceiling stairs bed couch table chair desk hallway closet
city: city town street road village downtown neighborhood avenue block corner park bridge highway york chicago london
country: country america american china england france india mexico africa europe nation world state states japan
vehicle: car cars truck bus train plane boat ship bike bicycle taxi cab subway airplane drove driving drive wheel
motion: walk walked run ran running go went come came move moved jump jumped climb climbed fall fell rush ride rode
travel: travel trip journey flight airport ticket vacation visit visited abroad arrived leave left return returned
speech: say said tell told ask asked talk talked speak spoke call called answer answered reply whisper explain explained
sound: sound noise loud quiet silence silent voice music song sing sang hear heard listen listened ring bang
vision: see saw seen look looked looking watch watched stare stared glance notice noticed eyes sight view appear
touch: touch touched feel felt hold held grab grabbed push pushed pull pulled press squeeze rub soft hard
mental: think thought know knew believe believed remember remembered forget forgot understand realize realized wonder idea mind imagine decide decided
money: money dollars dollar pay paid buy bought sell sold cost price cash rich poor bank job salary rent afford
work: work worked job boss office business company working career meeting employee manager project hired fired
school: school college university class teacher student students study studied learn learned homework exam grade professor lesson
law: police cop cops law lawyer judge court crime criminal jail prison arrest arrested steal stole guilty murder
violence: kill killed gun shot shoot hit beat attack attacked blood knife wound punch fight war bomb dead
death: die died death dead dying funeral grave kill killed buried body corpse cancer lost passed
health: doctor hospital sick ill disease medicine nurse patient surgery pill pills health cancer pain heal injury
religion: god church pray prayed prayer jesus christ heaven hell religious faith soul bible priest holy spirit
music: music song songs band play played guitar piano sing singing dance danced concert album drum
art: art paint painting picture photo photograph draw drew artist museum film movie movies book story write wrote
sport: game games play team ball football baseball basketball soccer win won lose lost score coach player race
clothing: shirt dress shoes hat coat jacket pants clothes wear wore wearing jeans suit boots shoe sock
household: phone bag box bottle key keys book books paper letter cup plate knife bed tv television computer camera
tool: tool hammer knife rope wire machine engine gun pen pencil needle scissors saw brush
tech: computer phone internet email online video screen camera radio television tv technology website text app
size: big small large little huge tiny tall short long giant enormous wide narrow thick thin
shape: round square circle line edge point corner flat curve straight shape ball ring
temperature: hot cold warm cool heat freezing burning boiling frozen fire burn burned ice
light: light dark darkness bright shadow shine shining lamp candle glow flash dim lit
quantity: all some many much more most few several every each lot lots none half whole enough plenty
social: friend friends party together group club meet met crowd community share help helped team neighbor
politics: government president election vote political party law congress senator power mayor king queen official
military: war army soldier soldiers military battle fight enemy weapon weapons gun troops navy officer camp
home: home house family bed sleep dinner kitchen garden yard porch couch living
sleep: sleep slept asleep dream dreamed dreams tired bed wake woke awake nap night rest
age: old young age older younger kid baby teenager adult elderly ancient new years born
gender: he she man woman boy girl male female men women his her lady guy sir
truth: true truth lie lied real really honest fact actually false wrong right correct
possession: have has had own owned got get give gave take took keep kept lose lost
existence: is was be been being exist exists there here live lived life alive
spatial: in on under over above below behind front inside outside near between beside around through across
direction: up down left right north south east west forward backward away toward towards back
speed: fast quick quickly slow slowly sudden suddenly immediately hurry rush soon instantly
texture: soft hard rough smooth wet dry sticky sharp heavy light warm
material: wood metal glass stone plastic paper iron steel gold silver cloth leather
water: water sea ocean river lake swim swam wave waves rain wet pool boat beach shore
fire: fire burn burned flame flames smoke smoking ash hot heat burning
emotion: feel felt feeling feelings emotion emotional mood heart soul care
desire: want wanted wish hope hoped need needed desire dream longing crave
ability: can could able ability skill try tried attempt manage managed succeed fail failed
change: change changed become became turn turned grow grew start started begin began end ended stop stopped
communication_obj: letter email message note phone call book news story word words sentence question
mind_state: confused sure certain doubt surprised shocked curious bored interested aware strange weird
morality: good bad evil wrong right fair kind cruel nice mean selfish generous
success: win won success successful fail failed lose lost best better worse worst achieve
power: power strong weak control force boss leader rule ruled authority
danger: danger dangerous risk safe safety careful threat warning accident crash emergency
time_rel: before after then later soon now already still yet again always never once
frequency: always never often sometimes usually rarely ever every daily constantly
evaluation: good bad great terrible awful nice beautiful ugly perfect amazing horrible wonderful
cognition_obj: idea plan reason problem question answer memory thought decision choice
abstract: life world time way thing things part place point fact case
body_action: sit sat stand stood lie lay sleep breathe breathed eat ate kneel lean
manipulation: open opened close closed put placed throw threw carry carried pick picked drop dropped
perception_other: smell smelled taste tasted sound feel felt hear heard
"""

CATEGORIES = []
for line in _CATS.strip().splitlines():
    name, words = line.split(':', 1)
    CATEGORIES.append(set(words.split()))
CATEGORIES = CATEGORIES[:72]

FUNCTION_WORDS = ['the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
                  'as', 'that', 'this', 'it', 'is', 'was', 'be', 'have', 'had', 'not', 'i', 'you', 'he', 'she', 'we',
                  'they', 'me', 'him', 'her', 'his', 'my', 'so', 'if', 'then', 'there', 'what', 'when', 'all', 'just',
                  'like', 'up', 'out', 'about', 'do', 'would']
FW_IDX = {w: i for i, w in enumerate(FUNCTION_WORDS)}

SUFFIXES = ['ing', 'ed', 's', 'ly', 'er', 'est', 'tion', 'ness', 'ment', 'able', 'ful', 'less']
PREFIXES = ['un', 're']

CLOSED = [
    {'i', 'me', 'my', 'mine', 'myself', "i'm", "i'd", "i've", "i'll"},
    {'we', 'us', 'our', 'ours', 'ourselves', "we're", "we've", "we'll"},
    {'you', 'your', 'yours', 'yourself', 'yourselves', "you're", "you've", "you'll"},
    {'he', 'him', 'his', 'himself', "he's"},
    {'she', 'her', 'hers', 'herself', "she's"},
    {'it', 'its', 'itself', "it's"},
    {'they', 'them', 'their', 'theirs', 'themselves', "they're"},
    {'what', 'who', 'whom', 'whose', 'which', 'where', 'when', 'why', 'how', 'whatever', 'whoever'},
    {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', "can't", "won't", "couldn't",
     "wouldn't", "shouldn't"},
    {'not', 'no', 'never', 'nothing', 'nobody', 'none', 'nowhere', 'neither', 'nor', "don't", "didn't", "doesn't",
     "isn't", "wasn't", "can't", "won't", "couldn't", "wouldn't", "aren't", "weren't", "haven't", "hasn't", "n't"},
    {'the', 'a', 'an'},
    {'this', 'that', 'these', 'those'},
    {'in', 'on', 'at', 'by', 'for', 'with', 'from', 'of', 'to', 'into', 'onto', 'over', 'under', 'about', 'through',
     'between', 'after', 'before', 'during', 'without', 'within', 'against', 'across', 'behind', 'around'},
    {'and', 'or', 'but', 'nor', 'yet', 'so'},
    {'because', 'although', 'though', 'while', 'if', 'unless', 'since', 'until', 'whether', 'whereas', 'once'},
    {'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being'},
    {'have', 'has', 'had', 'having', 'do', 'does', 'did', 'done'},
    {'all', 'some', 'any', 'many', 'much', 'few', 'several', 'every', 'each', 'most', 'more', 'less', 'both'},
    {'very', 'really', 'so', 'too', 'quite', 'extremely', 'pretty', 'totally', 'absolutely', 'just'},
]
PRONOUNS = set().union(*CLOSED[:7])

MENTAL_VERBS = {'think', 'thought', 'thinks', 'thinking', 'know', 'knew', 'known', 'knows', 'believe', 'believed',
                'remember', 'remembered', 'forget', 'forgot', 'realize', 'realized', 'wonder', 'wondered',
                'understand', 'understood', 'imagine', 'imagined', 'guess', 'suppose', 'feel', 'felt', 'decide',
                'decided', 'want', 'wanted', 'hope', 'hoped', 'mean', 'meant'}
DEIXIS_SPACE = {'here', 'there', 'this', 'that', 'these', 'those', 'come', 'came', 'go', 'went'}
DEIXIS_TIME = {'now', 'then', 'today', 'tomorrow', 'yesterday', 'tonight', 'ago', 'later', 'soon'}
DISCOURSE = {'well', 'so', 'oh', 'okay', 'ok', 'anyway', 'like', 'yeah', 'right', 'um', 'uh', 'actually',
             'basically', 'you know', 'however', 'besides', 'meanwhile', 'still', 'also', 'and'}

VOWELS = set('aeiouy')


def clean(tok):
    return re.sub(r"[^a-z']", '', tok.lower()).strip("'")


def syllables(w):
    w = w.replace("'", '')
    if not w:
        return 0
    groups = re.findall(r'[aeiouy]+', w)
    n = len(groups)
    if w.endswith('e') and n > 1 and not w.endswith('le'):
        n -= 1
    return max(n, 1)


IRREG_PAST = {'was', 'were', 'had', 'did', 'said', 'went', 'came', 'saw', 'got', 'made', 'took', 'knew', 'thought',
              'told', 'found', 'gave', 'felt', 'left', 'kept', 'ran', 'sat', 'stood', 'heard', 'brought', 'began',
              'wrote', 'ate', 'drank', 'fell', 'held', 'met', 'paid', 'sent', 'spoke', 'woke', 'became', 'lost',
              'put', 'let', 'meant', 'bought', 'drove', 'threw', 'won', 'wore', 'slept', 'caught', 'taught'}
PRESENT = {'is', 'are', 'am', 'do', 'does', 'has', 'have', "it's", "i'm", "don't", "doesn't", "isn't"}
FUTURE = {'will', "i'll", "you'll", "we'll", "he'll", "she'll", "they'll", "won't", 'gonna', 'shall'}

POS_TAGS = ['noun', 'verb', 'adj', 'adv', 'func', 'pron']


def guess_pos(w):
    if w in PRONOUNS:
        return 5
    if w in FW_IDX or any(w in c for c in CLOSED[7:]):
        return 4
    if w.endswith('ly'):
        return 3
    if w.endswith(('ing', 'ed', 'ize', 'ise', 'ate')) or w in IRREG_PAST:
        return 1
    if w.endswith(('ous', 'ful', 'less', 'able', 'ible', 'ive', 'al', 'ic', 'ish', 'est')):
        return 2
    return 0


def tense_cue(w):
    """0 past, 1 present, 2 future, 3 progressive, -1 none."""
    if w in FUTURE:
        return 2
    if w in IRREG_PAST or (w.endswith('ed') and len(w) > 3):
        return 0
    if w.endswith('ing') and len(w) > 4:
        return 3
    if w in PRESENT:
        return 1
    return -1


class WordFeats:
    def __init__(self):
        self.cache = {}
        self.dim = 45 + 45 + 2 + len(CATEGORIES) + 50 + 14 + len(CLOSED) + 6 + 4

    def _wordnet(self, w):
        h1 = np.zeros(45, np.float32)
        h2 = np.zeros(45, np.float32)
        rar, dep = 0.0, 0.0
        if wn is None or not w:
            return h1, h2, rar, dep
        try:
            syns = wn.synsets(w)
        except Exception:
            syns = []
        if not syns:
            return h1, h2, 1.0, 0.0
        for s in syns:
            i = LEX_IDX.get(s.lexname())
            if i is not None:
                h1[i] += 1
        h1 /= h1.sum() + 1e-9
        for s in syns[:3]:
            i = LEX_IDX.get(s.lexname())
            if i is None:
                continue
            h2[i] += 1 + len(s.lemmas())  # synset itself + its synonyms
            for hp in s.hypernyms():
                j = LEX_IDX.get(hp.lexname())
                if j is not None:
                    h2[j] += 1
            for dw in re.findall(r'[a-z]+', s.definition().lower()):
                if len(dw) < 4:
                    continue
                try:
                    ds = wn.synsets(dw)
                except Exception:
                    ds = []
                if ds:
                    j = LEX_IDX.get(ds[0].lexname())
                    if j is not None:
                        h2[j] += 0.5
        h2 /= h2.sum() + 1e-9
        cnt = 0
        for l in syns[0].lemmas():
            try:
                cnt += l.count()
            except Exception:
                pass
        rar = 1.0 / (1.0 + math.log1p(cnt) + math.log(len(syns)))
        try:
            dep = syns[0].min_depth() / 20.0
        except Exception:
            dep = 0.0
        return h1, h2, rar, dep

    def __call__(self, w):
        if w in self.cache:
            return self.cache[w]
        h1, h2, rar, dep = self._wordnet(w)
        cats = np.array([w in c for c in CATEGORIES], np.float32)
        fw = np.zeros(50, np.float32)
        if w in FW_IDX:
            fw[FW_IDX[w]] = 1
        aff = np.array([float(len(w) > len(s) + 2 and w.endswith(s)) for s in SUFFIXES] +
                       [float(len(w) > len(p) + 3 and w.startswith(p)) for p in PREFIXES], np.float32)
        cl = np.array([w in c for c in CLOSED], np.float32)
        L = max(len(w), 1)
        nv = sum(c in VOWELS for c in w)
        clusters = len(re.findall(r'[^aeiouy\']{2,}', w))
        phon = np.array([syllables(w) / 3.0, len(w) / 8.0, nv / L, (L - nv) / L, clusters / 2.0,
                         float(bool(w) and w[0] in VOWELS)], np.float32)
        misc = np.array([w in MENTAL_VERBS, w in DEIXIS_SPACE, w in DEIXIS_TIME, w in DISCOURSE], np.float32)
        v = np.concatenate([h1, h2, [rar, dep], cats, fw, aff, cl, phon, misc]).astype(np.float32)
        self.cache[w] = v
        return v


def entropy(counts):
    c = np.asarray(counts, np.float64)
    s = c.sum()
    if s <= 0:
        return 0.0
    p = c[c > 0] / s
    return float(-(p * np.log(p)).sum())


TAUS = [5, 20, 100, 500]


class Embedder:
    def __init__(self):
        self.wf = WordFeats()
        self.keep = None

    def features(self, texts):
        D = self.wf.dim
        N = len(texts)
        rows = []
        emas = [np.zeros(D, np.float64) for _ in TAUS]
        seen = set()
        novelty = 0.0
        since_sent, since_pron = 50, 50
        for t, text in enumerate(texts):
            raw = text.split()
            toks = [clean(x) for x in raw]
            n = len(toks)
            # near-uniform weighted 10-gram sum
            ssum = np.zeros(D, np.float32)
            tense = np.zeros(4, np.float32)
            pos_c = np.zeros(len(POS_TAGS))
            sup_c = np.zeros(45)
            for k, w in enumerate(toks):
                d = n - 1 - k
                wt = 1.0 - 0.02 * d
                v = self.wf(w)
                ssum += wt * v
                tc = tense_cue(w)
                if tc >= 0:
                    tense[tc] += 1
                if w:
                    pos_c[guess_pos(w)] += 1
                sup_c += v[:45]
            tense /= max(n, 1)
            qm = float(sum(x.count('?') for x in raw))
            em = float(sum(x.count('!') for x in raw))
            last_w = toks[-1] if toks else ''
            last = self.wf(last_w) if toks else np.zeros(D, np.float32)
            extras = np.concatenate([tense, [qm, em, entropy(pos_c) / 2.0, entropy(sup_c) / 3.0]])
            # story-level dynamics on newest word
            x = last.astype(np.float64)
            dist, cos = [], []
            for e in emas:
                dist.append(np.linalg.norm(x - e) / 3.0)
                ne = np.linalg.norm(e)
                nx = np.linalg.norm(x)
                cos.append(1.0 - (x @ e) / (nx * ne) if ne > 0 and nx > 0 else 1.0)
            for e, tau in zip(emas, TAUS):
                a = 1.0 / tau
                e *= (1 - a)
                e += a * x
            is_new = float(bool(last_w) and last_w not in seen)
            if last_w:
                seen.add(last_w)
            novelty = 0.95 * novelty + 0.05 * is_new
            # sentence end / pronoun counters
            last_raw = raw[-1] if raw else ''
            if last_w in PRONOUNS:
                since_pron = 0
            else:
                since_pron += 1
            dyn = np.concatenate([
                np.concatenate([e for e in emas]),
                emas[0] - emas[3],
                dist, cos,
                [math.log1p(since_sent), math.log1p(since_pron), novelty, is_new,
                 math.log1p(t) / 8.0, t / max(N - 1, 1)],
            ])
            if re.search(r'[.?!]["\')]*$', last_raw):
                since_sent = 0
            else:
                since_sent += 1
            rows.append(np.concatenate([ssum, last, extras, dyn]).astype(np.float32))
        return np.stack(rows) if rows else np.zeros((0, self.n_raw()), np.float32)

    def n_raw(self):
        D = self.wf.dim
        return 2 * D + 8 + 5 * D + 8 + 6

    def __call__(self, texts):
        X = self.features(texts)
        if self.keep is None:
            var = X.var(axis=0) if len(X) > 1 else np.ones(X.shape[1])
            self.keep = var >= 0.0022
            if self.keep.sum() == 0:
                self.keep[:] = True
        return np.nan_to_num(X[:, self.keep]).astype(np.float32)


def build_embedder():
    return Embedder()
