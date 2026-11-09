"""Untrained hand-wired one-layer transformer: semantic-category counts + recency-weighted hashed word identity."""
import hashlib
import numpy as np

MODEL_NAME = 'rec-densesem-v1'
MODEL_DESCRIPTION = ('Hand-wired 1-layer transformer, 1024 dims: char tokens + md5-hashed subword/word tokens '
                     '(Gaussian emb dims 563-1023) + ~70 semantic category tokens (8 dims each =1e4); 8 heads '
                     'with exponential recency decay over own 128-dim slice, head 0 uniform; residual last-word '
                     'hash; position dims; final LayerNorm.')

D = 1024
N_BUCKETS = 4096
HASH_START = 563
POS_DIMS = (560, 561, 562)
CAT_VAL = 10000.0
# per-token decay rates for heads 0..7 (head 0 = uniform mean); hash heads decay sharply
DECAYS = [0.0, 0.02, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]

LEX = {
 'ABSTRACT': 'idea ideas concept theory truth meaning reason purpose nature freedom justice principle fact facts thing things way kind sort reality existence value',
 'ACHIEVE': 'win won success succeed succeeded achieve achieved goal goals accomplish earned best champion victory prize award proud',
 'AFFECTION': 'love loved loves kiss kissed hug hugged dear darling sweetheart honey romance romantic adore tender cuddle',
 'ANIMAL': 'dog dogs cat cats horse horses bird birds fish cow cows pig pigs animal animals bear wolf mouse rat snake lion tiger chicken sheep deer duck rabbit puppy',
 'ANXIETY': 'afraid fear scared worried worry nervous anxious panic terrified frightened dread tense stress stressed',
 'ART': 'music song songs sing sang singing paint painting art artist dance danced dancing poem poetry film movie movies band guitar piano play',
 'BODY': 'hand hands head face eyes eye arm arms leg legs foot feet hair mouth heart body back finger fingers nose ear ears skin blood shoulder neck chest teeth knee stomach lips',
 'BUILDING': 'house home room rooms door doors window windows wall walls floor roof kitchen bedroom bathroom building apartment hall stairs office basement garage',
 'BUSINESS': 'money job work worked company business pay paid boss dollars bank buy bought sell sold price market office career hired fired salary',
 'CERTAINTY': 'always never sure certain definitely absolutely clearly obviously really totally completely exactly indeed',
 'CHANGE': 'change changed become became turn turned grow grew begin began start started end ended new',
 'CHILD': 'baby babies child children kid kids boy boys girl girls toddler infant young teenager',
 'CITY': 'city town street streets road roads car cars bus train subway downtown neighborhood village york london traffic',
 'CLOTHING': 'shirt shoes dress coat hat pants jacket clothes wear wore wearing jeans socks boots suit tie',
 'COGNITION': 'think thought know knew understand understood remember remembered believe believed realize realized wonder wondered mind decide decided imagine forget forgot learn learned',
 'COLOR': 'red blue green yellow black white color colors brown pink purple orange gray grey dark bright golden',
 'COMMUNICATION': 'say said says tell told talk talked talking ask asked answer answered call called speak spoke words word letter phone email question',
 'CONFLICT': 'fight fought war battle argue argued attack attacked enemy kill killed shoot shot gun guns hit army soldier',
 'DEATH': 'die died dead death dying funeral grave kill killed murder buried corpse ghost',
 'DRINK': 'drink drank water beer wine coffee tea juice milk bottle glass drunk alcohol',
 'EDUCATION': 'school class teacher student students college university study studied learn homework exam test professor lesson grade',
 'EMOTION': 'feel felt feeling feelings happy sad angry upset emotional cry cried tears laugh laughed smile smiled mood',
 'FAMILY': 'mother father mom dad mum brother sister son daughter family parents wife husband uncle aunt grandmother grandfather cousin grandma grandpa',
 'FEMALE': 'she her hers woman women girl lady mother sister daughter wife mrs queen',
 'FOOD': 'food eat ate eating dinner lunch breakfast bread meat cake pizza apple cook cooked meal hungry restaurant soup egg eggs',
 'FRIEND': 'friend friends buddy pal neighbor partner roommate guy guys',
 'FUTURE': 'will tomorrow future soon later next plan plans hope going someday',
 'GOVERNMENT': 'government president law police court judge state country nation political vote election congress officer king queen',
 'HEALTH': 'doctor hospital sick ill nurse medicine pain hurt cancer disease patient surgery healthy injury',
 'HEARING': 'hear heard listen listened sound sounds noise loud quiet voice voices silence',
 'HOME': 'home house bed couch table chair yard garden',
 'HUMOR': 'funny joke jokes laugh laughed laughing humor hilarious silly',
 'INSIGHT': 'realize realized understand understood discover discovered recognize figure figured',
 'LEISURE': 'party vacation game games play played fun holiday trip travel relax beach',
 'LOCATION': 'here there where place places inside outside behind front near far above below under over across around',
 'MALE': 'he him his man men guy boy father brother son husband mr sir king',
 'MOTION': 'go went gone walk walked run ran running move moved come came drive drove fly flew jump jumped climb climbed fall fell rush rushed leave left',
 'NATURE': 'tree trees sky sun moon star stars river lake ocean sea mountain mountains forest field grass flower flowers rain snow wind earth',
 'NEGATION': 'no not never nothing none nobody nowhere neither nor cannot dont didnt wont isnt wasnt',
 'NEGATIVE': 'bad terrible awful horrible wrong worse worst hate hated ugly nasty evil',
 'NUMBER': 'one two three four five six seven eight nine ten hundred thousand million first second third half dozen twenty',
 'PAST': 'was were had did ago yesterday before once used',
 'PEOPLE': 'people person everyone everybody someone somebody anyone crowd human humans folks',
 'PERCEPTION': 'see saw seen look looked looking watch watched notice noticed appear seemed seem',
 'POSITIVE': 'good great nice wonderful beautiful amazing perfect lovely fine best better awesome excellent',
 'POSSESSION': 'have has had own owned give gave take took get got keep kept',
 'POWER': 'power control strong leader lead led command rule rules force strength',
 'PRONOUN_I': 'i me my mine myself',
 'PRONOUN_THEY': 'they them their theirs themselves',
 'PRONOUN_WE': 'we us our ours ourselves',
 'PRONOUN_YOU': 'you your yours yourself',
 'QUANTITY': 'all some many much more most few little lot lots every each several enough',
 'QUESTION': 'what why how who when which whether',
 'RELIGION': 'god church pray prayed prayer heaven hell soul faith holy jesus bible spirit',
 'RISK': 'danger dangerous risk safe safety careful warning threat',
 'SADNESS': 'sad cry cried crying tears lonely grief miss missed sorry depressed lost',
 'SENSE_TOUCH': 'touch touched soft hard warm rough smooth hold held',
 'SEX': 'sex naked sexy lover',
 'SMELL_TASTE': 'smell smelled taste tasted sweet sour bitter',
 'SOCIAL': 'together meet met help helped share shared visit visited invite invited join joined',
 'SPACE': 'big small large tiny huge long short wide tall high low deep',
 'SPEECH': 'yes yeah okay oh um uh well like hey hello hi',
 'STILLNESS': 'stop stopped wait waited stay stayed sit sat stand stood',
 'TIME': 'time day days night nights morning evening week weeks month months year years hour hours minute minutes moment today tonight',
 'TOOL': 'knife hammer tool tools box key keys bag paper pen book books computer camera',
 'URGENCY': 'suddenly quickly immediately fast hurry finally instantly',
 'VEHICLE': 'truck plane boat ship bike bicycle taxi',
 'VIOLENCE': 'punch kick beat beaten stab violent',
 'WEATHER': 'storm cold hot sunny cloud clouds weather ice',
 'WORK': 'working jobs meeting project task employee',
}
CATS = sorted(LEX)[:70]
WORD2CATS = {}
for _ci, _c in enumerate(CATS):
    for _w in LEX[_c].split():
        WORD2CATS.setdefault(_w, []).append(_ci)


def _md5(s):
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % N_BUCKETS


def _stem(w):
    for suf in ('ing', 'ed', 'ly', 'es', 's'):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def _word_tokens(w):
    toks = ['len:%d' % min(len(w), 12), 'fl:' + w[0] + w[-1], 'sk:' + w[::2][:4], 'st:' + _stem(w)]
    b = '<' + w + '>'
    for n in range(2, 6):
        for i in range(len(b) - n + 1):
            toks.append('cg%d:' % n + b[i:i + n])
    toks.append('w:' + w)
    return toks


class Embedder:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.hash_emb = np.zeros((N_BUCKETS, D), dtype=np.float32)
        self.hash_emb[:, HASH_START:] = rng.standard_normal((N_BUCKETS, D - HASH_START)).astype(np.float32)
        self.cat_emb = np.zeros((len(CATS), D), dtype=np.float32)
        for ci in range(len(CATS)):
            self.cat_emb[ci, ci * 8:(ci + 1) * 8] = CAT_VAL
        self.cache = {}

    def _bucket(self, s):
        b = self.cache.get(s)
        if b is None:
            b = self.cache[s] = _md5(s)
        return b

    def _embed(self, text):
        words = text.lower().split()
        if not words:
            return np.zeros(D, dtype=np.float32)
        idx = [self._bucket('ch:' + ch) for ch in ''.join(words)]  # character tokens first
        rows = [self.hash_emb[i] for i in idx]
        last_word_vec = None
        for w in words:  # oldest to newest
            for t in _word_tokens(w):
                rows.append(self.hash_emb[self._bucket(t)])
            last_word_vec = rows[-1]
            key = ''.join(c for c in w if c.isalpha())
            for ci in WORD2CATS.get(key, []):
                rows.append(self.cat_emb[ci])
        X = np.stack(rows).copy()
        T = len(X)
        pos = np.arange(T, dtype=np.float32)
        X[:, POS_DIMS[0]] = pos / T
        X[:, POS_DIMS[1]] = np.sin(pos)
        X[:, POS_DIMS[2]] = np.cos(pos)
        age = T - 1 - pos
        out = np.zeros(D, dtype=np.float32)
        for h, lam in enumerate(DECAYS):
            a = np.exp(-lam * age)
            a /= a.sum()
            sl = slice(h * 128, (h + 1) * 128)
            out[sl] = a @ X[:, sl]
        out += last_word_vec  # residual: last word's identity hash
        out = (out - out.mean()) / (out.std() + 1e-5)  # LayerNorm, no affine
        return out

    def __call__(self, texts):
        return np.stack([self._embed(t) for t in texts]).astype(np.float32)


def build_embedder():
    return Embedder()
