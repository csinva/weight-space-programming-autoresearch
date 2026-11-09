"""Hand-written text features for predicting fMRI responses. Edit this file.

build_embedder() must return an object that maps a list of 10-gram strings (all of one story's
10-grams, in order) to a float array of shape (number of 10-grams, number of features).
"""
import re
import numpy as np
from nltk.corpus import wordnet as wn

MODEL_NAME = 'wnmorph-multitau-v1'
MODEL_DESCRIPTION = ('Untrained lexical features: WordNet lexname histograms (plain + extended with synonyms, '
                     'hypernyms, gloss words), synset rarity/depth, 72 semantic categories, 50 function words, '
                     '14 affix flags, 19 closed-class sets, crude phonology, mental/deixis/discourse lists; '
                     '10-gram near-uniform sum + last word + tense rates, ?/! and POS/supersense entropy; '
                     'story EMAs (tau 5,20,100,500), short-minus-long diff, distance/cosine surprise, log steps '
                     'since sentence end/pronoun, novelty rate, position; low-variance dims (<0.0022, first story) dropped.')

LEXNAMES = """adj.all adj.pert adv.all noun.Tops noun.act noun.animal noun.artifact noun.attribute noun.body
noun.cognition noun.communication noun.event noun.feeling noun.food noun.group noun.location noun.motive
noun.object noun.person noun.phenomenon noun.plant noun.possession noun.process noun.quantity noun.relation
noun.shape noun.state noun.substance noun.time verb.body verb.change verb.cognition verb.communication
verb.competition verb.consumption verb.contact verb.creation verb.emotion verb.motion verb.perception
verb.possession verb.social verb.stative verb.weather adj.ppl""".split()
LEX_IDX = {n: i for i, n in enumerate(LEXNAMES)}
NLEX = len(LEXNAMES)  # 45

# ---------------- 72 curated semantic categories ----------------
CATEGORIES = {
    'family': 'mother father mom dad mommy daddy sister brother son daughter wife husband aunt uncle cousin grandma grandmother grandfather grandpa parents parent family kids baby niece nephew',
    'friend_social': 'friend friends buddy pal neighbor stranger guy guys girl boy people person partner roommate colleague',
    'romance': 'love lover kiss kissed date dating boyfriend girlfriend married marriage wedding romance romantic sex crush divorce',
    'body_parts': 'head face eye eyes hand hands arm arms leg legs foot feet hair mouth nose ear ears finger fingers neck back shoulder chest stomach skin heart knee teeth lips',
    'body_function': 'breathe breath sleep slept eat ate drink sick pain hurt bleed blood sweat tired hungry pee cough',
    'health_medical': 'doctor hospital nurse medicine surgery cancer disease patient sick illness pill drug therapy clinic diagnosis',
    'death': 'die died dead death kill killed funeral grave murder buried corpse dying suicide',
    'emotion_pos': 'happy happiness joy glad love laugh laughed smile smiled excited proud fun wonderful great hope delighted',
    'emotion_neg': 'sad angry afraid fear scared upset cry cried hate worried anxious terrified lonely ashamed miserable depressed',
    'emotion_general': 'feel feeling felt emotion emotional mood heart nervous calm',
    'cognition': 'think thought know knew believe understand remember forget forgot realize realized idea mind wonder decide decided guess imagine',
    'perception_vision': 'see saw seen look looked looking watch watched stare stared notice noticed glance view sight eyes',
    'perception_sound': 'hear heard listen listened sound sounds noise loud quiet silence voice music song sing sang',
    'perception_touch': 'touch touched feel felt soft hard rough smooth warm cold hot wet',
    'speech': 'say said says tell told talk talked speak spoke ask asked answer answered call called yell yelled whisper shout',
    'writing_reading': 'write wrote written read book books letter letters story stories page paper word words note journal',
    'motion': 'go went gone come came walk walked run ran move moved drive drove fly flew climb jump jumped fall fell',
    'travel': 'trip travel traveled flight plane airport train bus car road highway journey ticket abroad',
    'vehicles': 'car cars truck bus bike bicycle boat ship plane train taxi cab vehicle',
    'home': 'house home room kitchen bedroom bathroom door window floor wall bed couch roof apartment garage yard',
    'furniture_objects': 'table chair desk bed couch lamp shelf box bag bottle cup glass plate key phone',
    'clothing': 'shirt pants dress shoes shoe hat coat jacket clothes wear wore wearing jeans socks suit uniform',
    'food': 'food eat ate dinner lunch breakfast meal bread meat chicken pizza cake coffee tea milk water apple sandwich cook cooked',
    'drink_alcohol': 'beer wine drink drinking drunk bar vodka whiskey alcohol cocktail',
    'animals': 'dog dogs cat cats horse bird birds fish cow pig animal animals mouse rat snake bear wolf lion',
    'nature': 'tree trees forest river lake ocean sea mountain mountains sky sun moon star stars field grass flower flowers rock',
    'weather': 'rain snow wind storm weather cloud clouds sunny cold hot heat ice thunder fog',
    'places_city': 'city town street streets village neighborhood downtown building park store shop restaurant church',
    'places_geo': 'america country state world china africa europe york california texas london paris mexico india',
    'school': 'school class teacher student students college university professor homework exam grade lesson campus study studied',
    'work_job': 'work worked job boss office career company business employee hire hired fired salary meeting',
    'money': 'money dollars dollar pay paid buy bought sell sold cost price cash rich poor bank afford',
    'time_units': 'day days night week weeks month months year years hour hours minute minutes second seconds morning evening',
    'time_relative': 'now then later before after soon already yet still ago today tomorrow yesterday tonight once',
    'numbers': 'one two three four five six seven eight nine ten hundred thousand million first second third half',
    'quantity': 'all some many much few more most less least every each several lot lots none enough',
    'size': 'big small large little huge tiny tall short long wide giant enormous',
    'color': 'red blue green yellow black white brown pink purple orange gray grey color dark bright',
    'spatial': 'up down in out over under above below inside outside near far behind front between around across top bottom side',
    'violence': 'fight fought hit punch shot shoot gun guns knife attack war soldier soldiers bomb violence beat',
    'crime_law': 'police cop cops arrest arrested jail prison crime criminal law lawyer court judge steal stole thief',
    'politics': 'government president politics political vote election party congress senator power nation country',
    'religion': 'god church pray prayer jesus christ religious faith heaven hell bible soul spirit holy',
    'technology': 'computer phone internet email online video camera screen machine technology software tv television radio',
    'sports_games': 'game games play played team ball football baseball basketball soccer win won lose lost score coach',
    'arts_music': 'music song songs band guitar piano art artist paint painting dance danced movie film show stage',
    'communication_media': 'news newspaper magazine radio show interview story article media message',
    'possession': 'have had has own owned give gave get got take took keep kept lose lost find found',
    'change': 'change changed become became turn turned grow grew start started begin began end ended stop stopped',
    'creation': 'make made build built create created draw drew fix fixed design',
    'contact_physical': 'hold held grab grabbed push pushed pull pulled throw threw catch caught touch hug hit',
    'social_actions': 'help helped meet met visit visited invite join party share thank apologize',
    'competition': 'win won lose lost beat compete competition race fight challenge',
    'consumption': 'eat ate drink drank smoke smoked swallow taste consume',
    'abstract': 'truth idea reason problem question answer fact thing things way kind sort nature',
    'evaluation_pos': 'good great nice best better beautiful amazing perfect wonderful awesome cool fine right',
    'evaluation_neg': 'bad worse worst terrible awful horrible wrong ugly stupid crazy weird strange',
    'certainty': 'sure certain definitely really actually exactly absolutely clearly obviously true',
    'uncertainty': 'maybe perhaps probably might possibly guess seem seemed unsure suppose',
    'intensifier': 'very really so too quite extremely totally completely pretty incredibly',
    'age_life': 'young old age child children kid kids baby teenager adult born birth grew life',
    'gender_people': 'man men woman women boy boys girl girls lady ladies guy gentleman',
    'occupation': 'doctor teacher lawyer nurse soldier officer driver worker artist writer farmer scientist manager',
    'sleep_dream': 'sleep slept asleep dream dreamed dreams wake woke awake bed tired nap',
    'light_dark': 'light dark darkness bright shadow lamp sun shine shining glow',
    'sound_events': 'bang crash knock ring rang scream screamed laugh noise clap',
    'desire_goal': 'want wanted need needed wish hope hoped try tried plan goal dream',
    'obligation': 'must should have need supposed ought required allowed',
    'achievement': 'succeed success fail failed failure achieve win won finish finished accomplish',
    'conflict': 'argue argued argument fight problem trouble conflict blame complain angry',
    'communication_tech': 'phone call called text texted email message letter wrote',
    'space_astronomy': 'space planet earth moon sun star stars universe rocket astronaut',
}
CAT_NAMES = list(CATEGORIES)
assert len(CAT_NAMES) == 72, len(CAT_NAMES)
CAT_SETS = [set(CATEGORIES[n].split()) for n in CAT_NAMES]

FUNCTION_WORDS = """the a an and but or so of to in on at for with from by as that this it is was be
not i you he she we they his her my there what if when just like all have had do""".split()
FUNCTION_WORDS = (FUNCTION_WORDS + ['me', 'were', 'are', 'would', 'about', 'up', 'out', 'then', 'one'])[:50]
assert len(FUNCTION_WORDS) == 50, len(FUNCTION_WORDS)
FW_IDX = {w: i for i, w in enumerate(FUNCTION_WORDS)}

SUFFIXES = ['ed', 'ing', 'ly', 's', 'er', 'est', 'tion', 'ness', 'ment', 'able', 'ful', 'less']
PREFIXES = ['un', 're']  # 14 affix flags in total

CLOSED = {
    'pron_1sg': 'i me my mine myself',
    'pron_1pl': 'we us our ours ourselves',
    'pron_2': 'you your yours yourself yourselves',
    'pron_3m': 'he him his himself',
    'pron_3f': 'she her hers herself',
    'pron_3n': 'it its itself',
    'pron_3pl': 'they them their theirs themselves',
    'wh': 'what who whom whose which where when why how whatever whoever',
    'modal': 'can could will would shall should may might must',
    'negation': "not no never nothing nobody none neither nor nowhere don't didn't can't won't isn't wasn't couldn't wouldn't doesn't",
    'article': 'a an the',
    'demonstrative': 'this that these those',
    'preposition': 'of in on at to for with from by about into over under through after before between against during without',
    'coord': 'and but or nor yet so',
    'subord': 'because although though since unless while whereas if whether until',
    'aux_be': "be am is are was were been being i'm he's she's it's we're they're you're",
    'aux_have': "have has had having i've we've they've you've",
    'aux_do': 'do does did doing done',
    'quantifier': 'all some any every each many much few several most both either',
}
CLOSED_NAMES = list(CLOSED)
assert len(CLOSED_NAMES) == 19
CLOSED_SETS = [set(CLOSED[n].split()) for n in CLOSED_NAMES]
PRONOUNS = set().union(*CLOSED_SETS[:7])
ALL_CLOSED = set().union(*CLOSED_SETS) | set(FUNCTION_WORDS)

MENTAL_VERBS = set('think thought know knew known believe believed remember remembered forget forgot realize realized understand understood wonder wondered imagine imagined feel felt want wanted hope hoped decide decided guess suppose figure figured'.split())
DEIXIS = set('here there now then this that these those today tomorrow yesterday tonight come came go went'.split())
DISCOURSE = set('well so anyway actually like okay ok oh um uh yeah right basically really you know'.split())

IRREG_PAST = set('was were had did said went came got saw knew thought told took made felt left found gave began ran sat stood heard kept brought held met wrote spoke sold bought caught taught fought lost won paid sent spent built became fell drove drew ate drank slept woke threw grew wore hid hit put set cut let'.split())
PRESENT_CUES = set("is am are does has do says goes gets i'm it's he's she's we're they're you're".split())
FUTURE_CUES = set("will shall gonna i'll you'll he'll she'll we'll they'll it'll won't".split())

VOWELS = set('aeiouy')
PLOSIVES = set('pbtdkgcq')
FRICATIVES = set('fvszhj')
NASALS = set('mn')
LIQUIDS = set('lrwy')

POS_CLASSES = ['func', 'noun', 'verb', 'adj', 'adv', 'other']
TAUS = [5, 20, 100, 500]


def clean(tok):
    return re.sub(r"[^a-z']", '', tok.lower()).strip("'")


def suffix_pos(w):
    if not w:
        return 5
    if w in ALL_CLOSED:
        return 0
    if w.endswith('ly'):
        return 4
    if w.endswith(('ing', 'ed')) or w in IRREG_PAST or w in MENTAL_VERBS:
        return 2
    if w.endswith(('ful', 'ous', 'able', 'ible', 'ive', 'al', 'ic', 'less', 'ish', 'est')):
        return 3
    if w.endswith(('tion', 'ness', 'ment', 'ity', 'er', 'or', 'ism', 'ist', 's', 'ship', 'hood')):
        return 1
    return 1


def tense_vec(w):
    past = float(w in IRREG_PAST or (w.endswith('ed') and len(w) > 3))
    pres = float(w in PRESENT_CUES)
    fut = float(w in FUTURE_CUES)
    prog = float(w.endswith('ing') and len(w) > 4)
    return np.array([past, pres, fut, prog])


def entropy(counts):
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts[counts > 0] / s
    return float(-(p * np.log(p)).sum())


class Embedder:
    def __init__(self):
        self.cache = {}
        self.first_lex = {}
        self.mask = None

    # ---------- WordNet helpers ----------
    def _first_lexname(self, w):
        if w not in self.first_lex:
            try:
                ss = wn.synsets(w)
            except Exception:
                ss = []
            self.first_lex[w] = LEX_IDX.get(ss[0].lexname()) if ss else None
        return self.first_lex[w]

    def _wordnet(self, w):
        h1 = np.zeros(NLEX)
        h2 = np.zeros(NLEX)
        extra = np.zeros(4)
        if not w:
            return h1, h2, extra
        try:
            ss = wn.synsets(w)
        except Exception:
            ss = []
        if not ss:
            return h1, h2, extra
        for s in ss:
            h1[LEX_IDX[s.lexname()]] += 1
        h2 += h1
        for s in ss[:3]:
            for lem in s.lemmas():
                n = lem.name().lower()
                if n != w and '_' not in n:
                    j = self._first_lexname(n)
                    if j is not None:
                        h2[j] += 0.5
            for hyp in s.hypernyms() + s.instance_hypernyms():
                h2[LEX_IDX[hyp.lexname()]] += 0.5
            for dw in re.findall(r'[a-z]+', s.definition().lower()):
                if len(dw) > 3 and dw not in ALL_CLOSED:
                    j = self._first_lexname(dw)
                    if j is not None:
                        h2[j] += 0.25
        h1 /= h1.sum()
        h2 /= max(h2.sum(), 1e-9)
        nsyn = len(ss)
        cnt = sum(l.count() for s in ss[:5] for l in s.lemmas() if l.name().lower() == w)
        depths = [s.min_depth() for s in ss[:3]]
        extra[0] = 1.0
        extra[1] = 1.0 / (1.0 + np.log1p(nsyn))           # rarity by polysemy
        extra[2] = 1.0 / (1.0 + np.log1p(cnt))            # rarity by tagged frequency
        extra[3] = np.mean(depths) / 15.0                  # depth
        return h1, h2, extra

    def word_vec(self, tok):
        if tok in self.cache:
            return self.cache[tok]
        w = clean(tok)
        h1, h2, wnx = self._wordnet(w)
        cats = np.array([float(w in s) for s in CAT_SETS])
        fw = np.zeros(len(FUNCTION_WORDS))
        if w in FW_IDX:
            fw[FW_IDX[w]] = 1.0
        aff = np.array([float(w.endswith(s) and len(w) > len(s) + 2) for s in SUFFIXES] +
                       [float(w.startswith(p) and len(w) > len(p) + 3) for p in PREFIXES])
        closed = np.array([float(w in s) for s in CLOSED_SETS])
        # crude phonology
        L = max(len(w), 1)
        syl = max(1, len(re.findall(r'[aeiouy]+', w))) if w else 0
        if w.endswith('e') and syl > 1 and not w.endswith(('le', 'ee')):
            syl -= 1
        phon = np.array([
            syl / 3.0, len(w) / 8.0,
            sum(c in VOWELS for c in w) / L,
            sum(c in PLOSIVES for c in w) / L,
            sum(c in FRICATIVES for c in w) / L,
            sum(c in NASALS for c in w) / L,
            sum(c in LIQUIDS for c in w) / L,
            float(bool(w) and w[0] in VOWELS),
            float(bool(w) and w[-1] in VOWELS),
            len(re.findall(r'[^aeiouy]{3,}', w)) / 2.0,
        ])
        lists = np.array([float(w in MENTAL_VERBS), float(w in DEIXIS), float(w in DISCOURSE)])
        v = np.concatenate([h1, h2, wnx, cats, fw, aff, closed, phon, lists])
        self.cache[tok] = (v, w)
        return self.cache[tok]

    def __call__(self, texts):
        n = len(texts)
        feats = []
        B = None
        emas = None
        seen = set()
        novelty = 0.0
        last_sent = None
        last_pron = None
        for i, text in enumerate(texts):
            toks = text.split()[-10:]
            if not toks:
                toks = ['']
            wvs = [self.word_vec(t) for t in toks]
            vecs = np.stack([x[0] for x in wvs])
            words = [x[1] for x in wvs]
            if B is None:
                B = vecs.shape[1]
                emas = [np.zeros(B) for _ in TAUS]
            k = len(toks)
            # near-uniform weights, slight recency emphasis
            wts = 0.97 ** np.arange(k - 1, -1, -1)
            ngram_sum = (wts[:, None] * vecs).sum(0)
            last = vecs[-1]
            tense = np.mean([tense_vec(w) for w in words], axis=0)
            q = sum(t.count('?') for t in toks)
            ex = sum(t.count('!') for t in toks)
            pos_counts = np.bincount([suffix_pos(w) for w in words], minlength=len(POS_CLASSES)).astype(float)
            ss_counts = vecs[:, :NLEX].sum(0)
            ngram_extra = np.array(list(tense) + [q, ex, entropy(pos_counts), entropy(ss_counts)])

            # story-level dynamics with the newest word
            surprise = []
            for e in emas:
                d = np.linalg.norm(last - e)
                ne, nl = np.linalg.norm(e), np.linalg.norm(last)
                cos = float(last @ e / (ne * nl)) if ne > 0 and nl > 0 else 0.0
                surprise += [d, 1.0 - cos]
            for j, tau in enumerate(TAUS):
                emas[j] = emas[j] + (last - emas[j]) / tau
            diff = emas[0] - emas[3]

            lw = words[-1]
            ltok = toks[-1]
            is_new = float(bool(lw) and lw not in seen)
            if lw:
                seen.add(lw)
            novelty = novelty + (is_new - novelty) / 20.0
            if lw in PRONOUNS:
                last_pron = i
            steps_sent = np.log1p(i - last_sent) if last_sent is not None else np.log1p(i + 1)
            steps_pron = np.log1p(i - last_pron) if last_pron is not None else np.log1p(i + 1)
            if re.search(r'[.!?]$', ltok):
                last_sent = i
            story = np.array(surprise + [steps_sent, steps_pron, novelty, np.log1p(i) / 8.0, i / 1000.0])

            feats.append(np.concatenate([ngram_sum, last, ngram_extra] + emas + [diff, story]))
        X = np.asarray(feats, dtype=np.float64)
        if self.mask is None:
            self.mask = X.var(0) >= 0.0022
            print(f'[embedder] total dims {X.shape[1]}, kept {int(self.mask.sum())}', flush=True)
        return np.nan_to_num(X[:, self.mask]).astype(np.float32)


def build_embedder():
    return Embedder()
