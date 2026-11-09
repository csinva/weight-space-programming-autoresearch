"""Hand-wired one-layer transformer: semantic-category counts + recency-weighted hashed word identity.

Token stream per 10-gram (lowercased, whitespace split): all character tokens first, then for each
word (oldest -> newest): length bucket, first+last letters, skip-bigram, crude stem, char 2-5-grams,
semantic-category tokens, whole-word identity. Hashed tokens (md5 -> 4096 buckets) have Gaussian
embeddings in dims 563-1023; category tokens set 8 dims per category (dims 3-562) to 10,000.
Dims 0-2: position value, token-type flag (char), constant 1.
8 heads, each pooling its own 128-dim slice with exponential recency decay (head 0 = uniform mean).
Residual adds the last token embedding; final LayerNorm (no affine).
"""
import hashlib
import numpy as np

MODEL_NAME = 'rec-densesem-v4'
MODEL_DESCRIPTION = ('Untrained 1-layer transformer: char + md5-hashed subword/word tokens (4096 buckets, Gaussian '
                     'dims 563-1023) + ~70 semantic-category tokens (8 dims each =1e4); 8 heads pooling own 128-dim '
                     'slice with exp recency decay (head0 uniform); residual last token; LayerNorm.')

D = 1024
HASH_LO = 563
N_BUCKETS = 4096
CAT_LO = 3
CAT_W = 8
CAT_VAL = 10000.0
N_HEADS = 8
HEAD_DIM = 128
# per-token decay rate for each head (head 0 = uniform mean)
DECAYS = [0.0, 0.005, 0.01, 0.02, 0.1, 0.3, 0.6, 1.0]

LEXICON = {
    'ABSTRACT': 'idea concept truth freedom justice meaning reason purpose theory nature reality existence fact sense quality value principle notion belief system',
    'ACTION': 'do did done doing make made making take took taken put get got give gave try tried use used work worked build built',
    'ANGER': 'angry anger mad furious rage hate hated yell yelled scream screamed fight fought annoyed upset frustrated',
    'ANIMAL': 'dog dogs cat cats horse horses bird birds fish cow cows pig pigs animal animals mouse rat snake bear lion wolf chicken deer',
    'ART': 'art music song songs paint painting picture photo film movie dance sing singing sang play band guitar piano poem book novel',
    'AUDITORY': 'hear heard hearing sound sounds listen listened noise loud quiet voice voices silence ring rang music echo',
    'BODY': 'body head face eye eyes hand hands arm arms leg legs foot feet hair heart mouth nose ear ears finger fingers skin back neck shoulder blood bone teeth',
    'BUILDING': 'house home room rooms door doors window windows wall walls floor roof kitchen bedroom building church school office apartment hall stairs',
    'CERTAINTY': 'sure certain definitely absolutely always never clearly obviously exactly really totally completely',
    'CHANGE': 'change changed become became turn turned grow grew start started stop stopped begin began end ended',
    'CITY': 'city town street streets road downtown york london chicago neighborhood block avenue village',
    'CLOTHING': 'shirt shoes dress pants hat coat jacket clothes wear wore wearing jeans suit boots glasses',
    'COGNITION': 'think thought thinking know knew known understand understood remember remembered forget forgot believe realize realized wonder wondered learn learned decide decided imagine',
    'COLD': 'cold ice snow winter freezing frozen cool chill',
    'COLOR': 'red blue green yellow black white brown pink purple orange gray grey color colors dark bright',
    'COMMUNICATION': 'say said says tell told talk talked speak spoke ask asked answer answered call called write wrote letter word words story phone',
    'DEATH': 'die died dead death dying kill killed funeral grave murder buried',
    'DRINK': 'drink drank water coffee tea beer wine bottle glass milk juice alcohol drunk',
    'EMOTION': 'feel felt feeling feelings love loved happy sad afraid scared fear worried excited proud sorry lonely hope',
    'FAMILY': 'mother father mom dad mommy daddy brother sister son daughter parents family wife husband baby kids children child grandmother grandfather aunt uncle cousin',
    'FOOD': 'food eat ate eating dinner lunch breakfast bread meat cake pizza cook cooked restaurant hungry meal apple',
    'FRIEND': 'friend friends buddy guy guys girl boy girlfriend boyfriend partner neighbor',
    'GOVERNMENT': 'government president law laws police court judge state country nation war army military vote election',
    'HEALTH': 'doctor hospital sick ill pain hurt nurse medicine cancer disease surgery health',
    'HOME': 'home bed couch table chair sofa kitchen garden yard',
    'HUMOR': 'funny laugh laughed laughing joke jokes smile smiled comedy',
    'LIGHT': 'light lights sun sunlight shine bright lamp shadow',
    'LOCATION': 'here there where place places somewhere everywhere around inside outside near far',
    'MONEY': 'money dollars dollar pay paid buy bought sell sold cost price rich poor bank job',
    'MOTION': 'go went gone going come came coming walk walked run ran move moved drive drove fly flew jump jumped climb fall fell leave left',
    'NATURE': 'tree trees forest river lake ocean sea mountain mountains sky field grass flower flowers rain wind beach island',
    'NEGATION': 'no not never nothing none nobody neither nor',
    'NUMBER': 'one two three four five six seven eight nine ten hundred thousand million first second third half number',
    'PERSON': 'man men woman women person people guy lady kid someone everyone somebody',
    'POSSESSION': 'have has had own owned keep kept belong mine yours',
    'RELIGION': 'god church pray prayer heaven hell jesus christ bible faith holy soul',
    'SCHOOL': 'school class teacher student students college university study studied homework grade',
    'SEXUALITY': 'sex kiss kissed sexy naked',
    'SIZE': 'big small little large huge tiny tall short long wide',
    'SOCIAL': 'together party meet met group team community crowd join',
    'SPACE': 'up down over under above below across through into onto between behind front top bottom side',
    'SPORT': 'game games ball team play played football baseball basketball soccer win won lose lost score',
    'TACTILE': 'touch touched feel hold held grab grabbed push pushed pull pulled soft hard rough smooth',
    'TECHNOLOGY': 'computer phone internet email radio tv television machine camera video',
    'TIME': 'time day days night year years week month morning evening hour minute minutes today tomorrow yesterday later before after soon then now when',
    'TRAVEL': 'travel trip car bus train plane airport flight ticket road hotel',
    'VEHICLE': 'car cars truck bus train plane boat ship bike bicycle',
    'VISUAL': 'see saw seen look looked looking watch watched stare stared eye notice noticed show showed',
    'WEATHER': 'weather rain storm sunny cloud clouds wind hot warm',
    'WORK': 'work job office boss business company worked working career',
    'QUANTITY': 'all some many much more most few lot lots every each enough',
    'PRONOUN_SELF': 'i me my myself mine',
    'PRONOUN_OTHER': 'he she him her his hers they them their you your we us our',
    'MODAL': 'can could would should might must may will shall',
    'QUESTION': 'what why how who which whom whose',
    'CAUSE': 'because so since therefore cause caused reason why',
    'SOCIAL_ROLE': 'king queen boss leader captain officer chief manager',
    'CONFLICT': 'fight war battle attack enemy argue argued',
    'DANGER': 'danger dangerous risk safe gun knife weapon bomb fire',
    'SLEEP': 'sleep slept asleep dream dreamed dreams wake woke tired bed',
    'MEMORY': 'remember remembered memory memories forget forgot past ago',
    'SURPRISE': 'surprise surprised suddenly shock shocked amazing wow',
    'POSITIVE': 'good great nice beautiful wonderful best better amazing perfect',
    'NEGATIVE': 'bad worse worst terrible awful horrible wrong ugly',
    'DESIRE': 'want wanted wish need needed desire like liked',
    'TRUTH': 'true truth lie lied lies honest real fake',
    'WATER': 'water swim swam river lake ocean sea wet wave waves',
    'YOUTH': 'young child kid kids teenager teen boy girl baby',
    'AGE': 'old older age aged elderly ancient',
}
_EXTRA = {
    'ABSTRACT': 'love life world way thing things problem situation power control chance point kind sort part history culture society experience knowledge mind spirit identity',
    'ACTION': 'doing does makes takes gets gives putting tries uses holding hold set sets setting opened open close closed break broke fix fixed',
    'ANGER': 'angrier hates yelling screaming pissed hostile shouted shout furious resent bitter',
    'ANIMAL': 'puppy kitten bug bugs bee spider monkey elephant tiger fox rabbit duck sheep goat insect pet pets whale shark',
    'ART': 'songs paintings artist artists stage theater show concert drawing drew draw camera poetry album record records sing singer',
    'AUDITORY': 'hears listening listen sounded noises loudly whisper whispered shouted shout heard bang knock knocked',
    'BODY': 'faces heads chest stomach belly lips tongue knees knee legs hips toes fingers throat brain lungs muscles tears breath breathing body',
    'BUILDING': 'houses rooms basement garage hallway bathroom elevator hotel tower store shop bar hospital prison castle cabin barn library',
    'CERTAINTY': 'certainly surely definitely course indeed must actually literally exactly absolutely',
    'CHANGE': 'changes changing becomes becoming turns turning grows growing starts starting stops begins ending different transform',
    'CITY': 'cities towns streets roads avenue downtown suburb suburbs manhattan brooklyn paris boston los angeles',
    'CLOTHING': 'dressed dresses shirts shoe hats coats sweater skirt uniform socks belt tie gloves scarf pocket',
    'COGNITION': 'thinks knows knowing understanding remembering believed believes realizing realize wondering learning deciding figured figure guess guessed mind idea ideas consider considered',
    'COLD': 'colder coldest freeze froze frost shiver shivering chilly icy',
    'COLOR': 'colored colorful golden silver gold darker blonde pale',
    'COMMUNICATION': 'saying tells telling talking speaking asking answering calling wrote writing letters message conversation explain explained discuss news shout reply replied',
    'DEATH': 'dies killing corpse body grave cemetery suicide passed lost',
    'DRINK': 'drinks drinking cup glasses vodka whiskey soda thirsty bar',
    'EMOTION': 'feels emotions happiness sadness joy cry cried crying tears nervous anxious upset embarrassed ashamed glad miss missed',
    'FAMILY': 'mothers fathers moms dads brothers sisters sons daughters kid grandma grandpa nephew niece relatives married marriage wedding',
    'FOOD': 'foods eats sandwich chicken rice soup cookies candy sugar fruit vegetables dessert kitchen ate',
    'FRIEND': 'friendship pal roommate classmate colleague mate',
    'GOVERNMENT': 'politics political senator congress mayor governor official officials federal soldiers soldier rights',
    'HEALTH': 'sickness illness injury injured wound ambulance pills drug drugs therapy patient',
    'LOCATION': 'away back somewhere anywhere nowhere corner middle edge center',
    'MONEY': 'cash paying spend spent afford earn earned wealthy salary rent debt',
    'MOTION': 'goes walks walking running runs moving drove driving flying jumping falling fallen leaving arrive arrived follow followed rush rushed',
    'NATURE': 'woods hill hills valley desert stars moon earth land stone rock rocks dirt',
    'NUMBER': 'twenty thirty forty fifty dozen once twice numbers',
    'PERSON': 'persons human humans folks stranger strangers individual everybody anybody nobody',
    'SCHOOL': 'classes teachers classroom lesson lessons exam test campus professor education',
    'SPACE': 'out off upon toward towards beneath beside inside outside near far along',
    'TIME': 'times moment moments days nights years weeks months hours seconds early late ago while until during always forever',
    'VISUAL': 'sees seeing looks watching watches glance glanced view saw noticed appear appeared appearance',
    'WORK': 'jobs employee employees hired fired meeting project',
}
for _c, _ws in _EXTRA.items():
    LEXICON[_c] = LEXICON[_c] + ' ' + _ws
CATS = sorted(LEXICON)
WORD2CATS = {}
for _ci, _c in enumerate(CATS):
    for _w in LEXICON[_c].split():
        WORD2CATS.setdefault(_w, [])
        if _ci not in WORD2CATS[_w]:
            WORD2CATS[_w].append(_ci)
assert CAT_LO + CAT_W * len(CATS) <= HASH_LO


def _bucket(s):
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16) % N_BUCKETS


def _stem(w):
    for suf in ('ing', 'ed', 'es', 's', 'ly', 'er'):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def _word_tokens(w):
    """Token ids for one word: hashed ids (>=0) and category ids encoded as -(1+ci)."""
    toks = [_bucket('len:%d' % min(len(w), 12)),
            _bucket('fl:%s%s' % (w[0], w[-1])),
            _bucket('skip:%s' % (w[0] + w[2] if len(w) > 2 else w)),
            _bucket('stem:' + _stem(w))]
    for n in range(2, 6):
        for i in range(len(w) - n + 1):
            toks.append(_bucket('cg:' + w[i:i + n]))
    for ci in WORD2CATS.get(w, []):
        toks.append(-(1 + ci))
    toks.append(_bucket('w:' + w))
    return toks


class Embedder:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.hash_emb = rng.standard_normal((N_BUCKETS, D - HASH_LO)).astype(np.float64)
        self.char_emb = {}
        self._wcache = {}

    def _char_tok(self, c):
        return _bucket('ch:' + c)

    def _tokens(self, text):
        words = text.lower().split()
        chars = [self._char_tok(c) for c in ''.join(words)]
        toks, is_char = list(chars), [1.0] * len(chars)
        for w in words:
            if w not in self._wcache:
                self._wcache[w] = _word_tokens(w)
            wt = self._wcache[w]
            toks += wt
            is_char += [0.0] * len(wt)
        return toks, np.array(is_char)

    def _embed(self, toks, is_char):
        T = len(toks)
        E = np.zeros((T, D))
        E[:, 0] = np.arange(T)  # position value
        E[:, 1] = is_char
        E[:, 2] = 1.0
        toks = np.array(toks)
        hm = toks >= 0
        E[hm, HASH_LO:] = self.hash_emb[toks[hm]]
        for t in np.where(~hm)[0]:
            ci = -toks[t] - 1
            E[t, CAT_LO + CAT_W * ci: CAT_LO + CAT_W * (ci + 1)] = CAT_VAL
        return E

    def __call__(self, texts):
        out = np.zeros((len(texts), D), dtype=np.float32)
        for i, text in enumerate(texts):
            toks, is_char = self._tokens(text)
            if not toks:
                continue
            E = self._embed(toks, is_char)
            T = len(toks)
            age = (T - 1 - np.arange(T)).astype(np.float64)
            x = E[-1].copy()  # residual: last token (last word's identity hash)
            for h in range(N_HEADS):
                w = np.exp(-DECAYS[h] * age)
                w /= w.sum()
                sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                x[sl] += w @ E[:, sl]
            x = (x - x.mean()) / (x.std() + 1e-5)  # LayerNorm, no affine
            out[i] = x
        return out


def build_embedder():
    return Embedder()
