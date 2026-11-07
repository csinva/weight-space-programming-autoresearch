"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Iter 21 — 2-layer bigram-detection circuit.

    Layer 0: shift attention copies prev-char into residual via quadratic
        positional features; MLP fires 60 hyp-bigram detectors as
        ReLU(curr[c2] + prev[c1] - 1).
    Layer 1: uniform-over-hyp attention at LAST_POS counts both char and
        bigram occurrences in the hypothesis.
    Head: in-sample multinomial-LR coefficients over (28 hyp char counts,
        60 hyp bigram counts).
    """
    torch.manual_seed(0)

    for block in model.blocks:
        block.ln1 = nn.Identity()
        block.ln2 = nn.Identity()
    model.final_ln = nn.Identity()

    id_0 = task.stoi["0"]
    id_1 = task.stoi["1"]
    id_2 = task.stoi["2"]

    HYP_START = task.SENT_LEN + 1
    HYP_END   = 2 * task.SENT_LEN + 1
    LAST_POS  = task.prompt_len - 1
    HYP_LEN   = HYP_END - HYP_START

    CHARS = "abcdefghijklmnopqrstuvwxyz _"
    assert len(CHARS) == 28
    char_idx = {c: i for i, c in enumerate(CHARS)}

    # Residual layout (d_model = 224)
    IS_HYP_DIM    = 28
    IS_LAST_DIM   = 29
    POS_LIN_DIM   = 30
    POS_QUAD_DIM  = 31
    POS_CONST_DIM = 32
    PREV_BASE     = 33   # ..60   (28 prev-char dims)
    BIG_BASE      = 61   # ..210  (150 bigram-detector dims)
    HCOUNT_BASE   = 211  # ..238  (28 hyp char counts)
    BCOUNT_BASE   = 239  # ..388  (150 hyp bigram counts)

    BIGRAMS = ['__', 'e ', 'in', ' a', 's ', 'n ', 'a ', 'th', 'he', 'ng', 'g ', ' i', 'an', ' t', 're', 'is', ' w', 'ar', ' s', 'er', ' p', 'ma', ' o', ' m', ' b', ' c', 'le', 'on', 'r ', ' h', 't ', 'wo', 'pl', 'd ', 'at', 'om', 'hi', 'pe', ' f', 'ou', ' d', 'it', 'y ', 'ea', 'o ', 'to', 'en', 'or', 'me', 'ch', ' g', 'st', 'ri', 'al', 'op', 'nd', 'e_', 'ro', 'ti', 'si', 'wa', 'la', 'es', 'te', 'il', ' r', 'rs', 'eo', 'ir', 'ki', 'ta', 'do', 'ee', 'de', 's_', 'ld', 'et', 'di', ' l', 'ne', 'as', 'ho', 'oo', 'ow', 'ts', 'ha', 'l ', 'ed', 'co', 'll', 'ay', 'un', 'ra', 'nt', 'g_', 'h ', 'id', 'tr', 'fo', 'of', 'so', 'bo', 'ke', 'wi', 'lo', 'ic', 'ai', 'f ', 'ac', 'ut', 'ot', 'og', 'ca', 'se', 'yi', 'li', 'r_', 't_', 'n_', 'ce', 'pi', 'up', 'el', 'ol', 'sh', 'ba', 'pa', 'lk', 'we', 'gr', 'ur', 'ck', 'ss', ' e', 'tw', 'am', 'd_', 'gi', 'rt', 'ad', 'be', 'p ', 'tt', 'dr', ' n', 'k ', 'ge', 'bi', 'fi', 'mi']
    K = len(BIGRAMS)
    assert K == 150
    CHAR_COEF = [[0.05624333498333393, 0.020349070023728072, -0.0032589284215395566, 0.1307484948674229, 0.08150859497935056, -0.17778482588965835, 0.002692540084342661, -0.00867516594576632, 0.042145896855562866, 0.14852967400029443, 0.25970027646792054, -0.003013701589029678, 0.03943103381639512, -0.16416694031743215, -0.13891073672782836, -0.06074198662244461, -0.34994813115699863, -0.17750068741232167, -0.10062344571079844, -0.11151455776743575, 0.006891708451057645, -0.055884113291753836, -0.04106667871474317, 0.12456831461079701, -0.0654197101383429, -0.365518000650462, -0.11356160097164693, -0.3912244347351316], [0.11450454334595066, -0.25909743481366737, -0.012834416498782517, 0.004186720012424217, -0.13270378502325875, 0.005444301636918294, 0.04923086017104446, 0.1347989103454996, 0.32963969099103957, 0.29491959860161177, -0.06939261203559179, -0.17652036405346253, -0.04490989047129671, -0.29611588073371337, 0.09082907787782901, 0.06722235597991309, 0.0536910148332582, 0.20735453903171255, 0.07333548218894353, 0.051504938187692834, 0.13970309651034535, -0.06993398067566428, 0.05186008050776617, 0.30184999614474517, 0.23578053957106623, -0.1869662796267117, 0.07011441777868582, 0.10170743932731469], [-0.1707478783293672, 0.2387483647899759, 0.016093344920326016, -0.13493521487985707, 0.05119519004392363, 0.17234052425291602, -0.05192340025536859, -0.12612374439967836, -0.3717855878467991, -0.4434492726021542, -0.19030766443243935, 0.17953406564261687, 0.005478856654844649, 0.4602828210507806, 0.048081658850199376, -0.006480369357454294, 0.29625711632376445, -0.029853851619171458, 0.027287963521913216, 0.06000961957972233, -0.1465948049614004, 0.12581809396741914, -0.010793401793017717, -0.4264183107555074, -0.17036082943265454, 0.5524842802767557, 0.043447183193027226, 0.2895169954078411]]
    BIGRAM_COEF = [[0.4161690659763974, -0.20547699298604238, -0.17437906397753153, 0.001538224953875177, 0.06227845411781101, 0.0906258621043714, 0.2749647670894049, 0.11096379620063396, -0.028729566493176897, 0.16738365682388734, 0.03703743858756271, 0.16344995299612283, 0.112661076321294, 0.09388503068468806, 0.3293290835295731, -0.08854630445852252, 0.35262128762199535, 0.12340529135596794, -0.08163646434152266, 0.27187121478848864, -0.10013463812977176, 0.14261599488397045, 0.2822236002163683, -0.17501864935056533, -0.027973798914345656, -0.23379617489681348, -0.033880282766505315, 0.007789550138528095, -0.019244043341513608, -0.14484382555503578, -0.1745339455939273, -0.20285289838974524, 0.09428842726338024, 0.13935729592867843, 0.10556265342387258, -0.16401283736576136, 0.11276312000542026, 0.1938121280235664, -0.15565384157809575, 0.019764594501619673, -0.28232700882399003, 0.36545616993276214, 0.07050978856108449, -0.31518215916435643, 0.01753788727173729, 0.198614424222306, -0.13990788601466464, 0.213556736481897, 0.2781565914684942, -0.24105566427182143, 0.05261580368945707, 0.40005836302138237, -0.043753543576955, 0.20945174824104124, 0.2260557627482736, -0.017826214627461756, 0.11488814780260447, 0.3337296376541101, -0.10690857985768952, -0.0338425808742538, 0.0019802020170591184, -0.15323477208313144, -0.13815626861833238, -0.06104155856196841, 0.4302564436449489, -0.2515664207342714, 0.13340191704481982, 0.35222844919270346, 0.1277220421847357, -0.3032076423090571, -0.05378183176259344, 0.15867568293057116, -0.02185289772263191, -0.09753122741514057, 0.22875644630725386, -0.006555072132369796, -0.07831181353169246, -0.05676496418942709, -0.13428034815214882, 0.13136271826216833, 0.042499494188980184, 0.03746698153844225, 0.09059144287078705, 0.23831422872780084, -0.11608665855040091, 0.13763930446258277, -0.0028134117232061578, -0.4294835398636931, 0.2981841517856363, -0.31027194934966373, 0.11642058448810563, 0.18099872818888504, 0.02931592037792413, 0.5664434210209891, 0.33207949407344234, -0.22744646139880506, 0.1488123245847867, 0.005441753425450626, 0.17259872288654268, -0.04647962684499186, 0.5929892823221649, 0.09020029196867588, -0.2650709688311574, -0.23772455536408346, 0.3474402728389829, 0.13801772309722515, -0.03575897178995019, 0.5627981488449397, 0.34261174722798937, 0.6593434334558567, 0.31244663827251934, 0.137151488646384, -0.1685405864082683, 0.4155300803965353, 0.06715224945216032, -0.17966380266675142, 0.463055654963936, 0.13588048196643346, 0.30355941134220676, -0.08950334196427234, -0.3004787806474479, 0.18187633062547806, -0.20075506425785059, 0.38423539866986595, 0.11295480315255292, -0.17726823231735048, -0.1865705149289836, -0.35778926357852503, -0.16232052281895196, 0.01803127765408482, 0.4013922230585903, -0.1211225367702146, 0.4409164430655585, -0.31179273111394806, 0.6614856648369803, -0.023079534493639163, 0.08189165369488006, 0.030212624400691824, 0.4337036402144173, 0.2986146529559273, 0.2677534302176408, -0.2113670746882546, -0.02693748456182467, 0.0758651534661695, 0.07689301735766071, -0.10634895046636923, -0.2076143672649471, -0.09603219953229285, 0.20951207477943384, -0.3947557233126413], [-0.11940263342830908, 0.10826691661490463, 0.13883391240852253, -0.004410929717454703, -0.16449318605126903, 0.009622190317555979, -0.20921940252023064, -0.28599857270854034, 0.257871625933699, -0.07237426494457024, 0.00593999567144352, -0.42604097424112153, 0.078761544994041, 0.022853824677976007, -0.06792362950931595, 0.18136399794456956, -0.10644126906983865, -0.1673252452542126, 0.07062577776377174, -0.1438645915505845, 0.02045553638643984, -0.04548343079924526, -0.2511099407726776, 0.07101941204573006, 0.08924663442405269, -0.03872073061307339, 0.10670077689500453, 0.3258647427136648, -0.002998085150716522, 0.020467595302865616, 0.054247470316442455, -0.08810848119287396, 0.01395557252236025, -0.030925094320616166, -0.2785163065116282, -0.022256078079378204, -0.40299963611108847, -0.04053818450473586, 0.23794419797617922, -0.08135012848730805, -0.009250387514295738, -0.3422537835357258, -0.11357000420906457, 0.08177268931900139, 0.03932419314896678, 0.06770745337568528, 0.5098205893961125, 0.028459470218754723, -0.06461730659205332, 0.21151198390691225, -0.20533111325126763, -0.225814609796567, -0.320749480950819, -0.005035739156592388, -0.003416440945080239, 0.1758239457124709, 0.0728041048420896, -0.2999721846304645, -0.00835621652432419, -0.36847460412306465, -0.2145087518625384, 0.28316552397765093, 0.22364973729468637, 0.1786283760840775, -0.14014614964791416, 0.0020808184501909348, -0.16707126912307824, -0.2664138100525474, -0.3279556044942532, -0.03467873019486291, 0.02723000836868266, -0.23740693504201488, -0.16914545357045896, 0.12744272084615899, -0.27687248430607414, -0.010589945004375061, 0.255545326318848, -0.26407425107103394, 0.2895889374895254, 0.0807596029315028, -0.11729943532571115, -0.11666681292792519, -0.17128669702799745, 0.13173911161528076, 0.13779137702987587, 0.1470867751018964, 0.26626899440703566, 0.05782400824323025, 0.133097854288436, 0.2603461029308915, -0.5865282815537144, -0.17033949232874068, -0.14017467549188434, -0.21105773974496628, -0.08819588666930626, 0.0729664532182404, -0.4890702003237826, -0.1320143888836853, 0.06702593084617124, -0.11376987536924583, -0.0557930719250035, -0.13246229176006075, -0.1338772221835411, -0.00915800966553794, -0.19746151255101113, -0.13917790152918855, -0.24378269003080555, -0.04490555364319185, -0.1698603596259668, -0.011196123093535867, -0.15419972705976584, 0.06248594043925381, 0.17316480113818106, -0.25058650254280773, -0.11254641944422934, -0.14928234679791744, -0.4745258803846141, -0.1258417583914017, 0.06473125786808467, 0.04921336350339196, -0.5026162224952282, 0.08140528465666572, 0.0833069031831456, -0.02283323930816054, -0.08402567631174591, 0.034137072501724264, 0.21992813855172286, 0.4180759359322308, 0.22892136282424458, 0.011732919279710699, -0.32109272632433805, -0.16906541226792393, 0.19365556508946907, -0.11738613397723086, -0.2647443436627771, -0.0985941915765426, 0.02501438975032411, 0.08884816362596001, -0.025269595806157525, -0.4175315862802469, -0.033456081040875135, -0.018727199233090845, -0.08708022069768176, -0.36693099273958774, 0.07689543629772157, 0.09146508183886257, 0.15407494959731086, 0.1855621156090615, -0.4082140688322055, 0.024441689422564725], [-0.29676643254768353, 0.09721007637101874, 0.03554515156902402, 0.002872704763578832, 0.10221473193336245, -0.10024805242193353, -0.06574536456947298, 0.17503477650786298, -0.2291420594405896, -0.09500939187924601, -0.04297743425901608, 0.2625910212451532, -0.1914226213154123, -0.11673885536271725, -0.26140545402030324, -0.09281769348595412, -0.246180018551928, 0.043919953898420974, 0.011010686577771978, -0.12800662323803752, 0.07967910174320202, -0.09713256408464022, -0.03111365944377948, 0.1039992373048835, -0.06127283550972792, 0.27251690550991137, -0.07282049412850859, -0.33365429285223097, 0.022242128492236005, 0.12437623025210873, 0.12028647527754274, 0.29096137958287793, -0.10824399978577144, -0.108432201607921, 0.17295365308764402, 0.18626891544509183, 0.2902365161057943, -0.15327394351891568, -0.08229035639798346, 0.0615855339856301, 0.291577396338164, -0.02320238639718544, 0.04306021564805129, 0.23340946984495878, -0.05686208042074609, -0.26632187759810405, -0.36991270338128945, -0.24201620670062848, -0.21353928487634047, 0.02954368036487513, 0.15271530956181537, -0.17424375322475666, 0.36450302452773575, -0.20441600908425311, -0.22263932180319723, -0.15799773108498175, -0.187692252644782, -0.033757453023563797, 0.11526479638199673, 0.40231718499723457, 0.21252854984551323, -0.12993075189471845, -0.0854934686763699, -0.11758681752203151, -0.2901102939970694, 0.24948560228401961, 0.03366935207840227, -0.08581463914049023, 0.20023356230950135, 0.33788637250386333, 0.026551823393881888, 0.0787312521116905, 0.19099835129304746, -0.029911493430987833, 0.04811603799911446, 0.017145017136747213, -0.17723351278708777, 0.32083921526010306, -0.15530858933747366, -0.21212232119386815, 0.07479994113675395, 0.07919983138953729, 0.080695254157112, -0.37005334034302806, -0.021704718479521403, -0.2847260795644302, -0.2634555826838536, 0.3716595316202268, -0.4312820060743488, 0.04992584641861136, 0.47010769706525607, -0.0106592358601823, 0.110858755113962, -0.35538568127567466, -0.24388360740398557, 0.15448000818042135, 0.34025787573914273, 0.1265726354583932, -0.23962465373261355, 0.16024950221424805, -0.537196210397273, 0.04226199979142984, 0.39894819101493084, 0.24688256502955738, -0.14997876028789783, 0.0011601784320375889, 0.279541661820686, -0.5178925952023552, -0.17275138760181794, -0.648147310361987, -0.1582469112126192, -0.19963742908553253, -0.00462421472999073, -0.16494357785377692, 0.045394169992137176, 0.32894614946465167, 0.01147022542088606, -0.010038723575032577, -0.3682906692105864, 0.040289978460845595, 0.8030950031430132, -0.26328161528185484, 0.11744816107476926, -0.3614021593618017, -0.028929126840837024, 0.14313115981567934, -0.033357623622835515, -0.06028667235421487, -0.06660084000551948, -0.029764196933778124, -0.08029949673429217, 0.290187949038388, -0.6345720081543834, 0.4291788650908411, -0.3967413211744392, 0.12167372607028311, -0.1069060434453062, -0.11906078802667448, -0.40843404440812114, 0.11891693332401118, -0.23429734917683456, 0.23009427392138818, 0.11401770525944803, 0.29106583927350876, -0.1537884536553523, 0.014883868627548553, 0.053539417667645334, -0.08952991607675923, 0.1987019940527006, 0.3703140338902643]]
    INTERCEPT = [-0.1930677484829899, -0.028123199281948185, 0.22119094776512768]

    # ---- Token embedding: char one-hot to dims 0..27 ----
    nn.init.zeros_(model.token_emb.weight)
    for i, c in enumerate(CHARS):
        if c in task.stoi:
            model.token_emb.weight.data[task.stoi[c], i] = 1.0

    # ---- Position embedding: hyp/last flags + quadratic pos features ----
    nn.init.zeros_(model.pos_emb.weight)
    for i in range(HYP_START, HYP_END):
        model.pos_emb.weight.data[i, IS_HYP_DIM] = 1.0
    model.pos_emb.weight.data[LAST_POS, IS_LAST_DIM] = 1.0
    for i in range(model.max_seq_len):
        model.pos_emb.weight.data[i, POS_LIN_DIM]   = float(i)
        model.pos_emb.weight.data[i, POS_QUAD_DIM]  = float(i * i)
        model.pos_emb.weight.data[i, POS_CONST_DIM] = 1.0

    # =====================================================================
    # Layer 0: shift attention (q.k peaks at j=i-1) + bigram MLP
    # =====================================================================
    attn0 = model.blocks[0].attn
    nn.init.zeros_(attn0.W_q.weight)
    nn.init.zeros_(attn0.W_k.weight)
    nn.init.zeros_(attn0.W_v.weight)
    nn.init.zeros_(attn0.W_o.weight)

    # q.k = -SCALE * (j - (i-1))**2  using:
    #   Q[i] = [SCALE*1, SCALE*2*(i-1), -SCALE*(i-1)**2]  (3 dims)
    #   K[j] = [-j**2, j, 1]                              (3 dims)
    # Expand (i-1)**2 = i**2 - 2*i + 1.
    SHIFT_SCALE = 100.0
    # Q row 0 = SHIFT_SCALE * pos_const
    attn0.W_q.weight.data[0, POS_CONST_DIM] = SHIFT_SCALE
    # Q row 1 = SHIFT_SCALE * 2*(i-1) = 2*SHIFT_SCALE*pos_lin - 2*SHIFT_SCALE*pos_const
    attn0.W_q.weight.data[1, POS_LIN_DIM]   = 2.0 * SHIFT_SCALE
    attn0.W_q.weight.data[1, POS_CONST_DIM] = -2.0 * SHIFT_SCALE
    # Q row 2 = -SHIFT_SCALE*(i**2 - 2i + 1) = -SHIFT_SCALE*pos_quad + 2*SHIFT_SCALE*pos_lin - SHIFT_SCALE*pos_const
    attn0.W_q.weight.data[2, POS_QUAD_DIM]  = -SHIFT_SCALE
    attn0.W_q.weight.data[2, POS_LIN_DIM]   = 2.0 * SHIFT_SCALE
    attn0.W_q.weight.data[2, POS_CONST_DIM] = -SHIFT_SCALE
    # K row 0 = -j**2
    attn0.W_k.weight.data[0, POS_QUAD_DIM]  = -1.0
    # K row 1 = j
    attn0.W_k.weight.data[1, POS_LIN_DIM]   = 1.0
    # K row 2 = 1
    attn0.W_k.weight.data[2, POS_CONST_DIM] = 1.0

    # V copies char one-hot (dims 0..27) directly
    for d in range(28):
        attn0.W_v.weight.data[d, d] = 1.0
    # W_o routes attn-out dims 0..27 to PREV_BASE..PREV_BASE+27
    for d in range(28):
        attn0.W_o.weight.data[PREV_BASE + d, d] = 1.0

    # ---- Layer 0 MLP: bigram detectors ReLU(curr[c2] + prev[c1] - 1) ----
    mlp0 = model.blocks[0].mlp
    nn.init.zeros_(mlp0.fc1.weight); nn.init.zeros_(mlp0.fc1.bias)
    nn.init.zeros_(mlp0.fc2.weight); nn.init.zeros_(mlp0.fc2.bias)
    for k, big in enumerate(BIGRAMS):
        c1, c2 = big[0], big[1]
        mlp0.fc1.weight.data[k, char_idx[c2]]             = 1.0
        mlp0.fc1.weight.data[k, PREV_BASE + char_idx[c1]] = 1.0
        mlp0.fc1.bias.data[k] = -1.0
        mlp0.fc2.weight.data[BIG_BASE + k, k] = 1.0

    # =====================================================================
    # Layer 1: uniform-over-hyp attention at LAST_POS counts chars+bigrams
    # =====================================================================
    attn1 = model.blocks[1].attn
    nn.init.zeros_(attn1.W_q.weight)
    nn.init.zeros_(attn1.W_k.weight)
    nn.init.zeros_(attn1.W_v.weight)
    nn.init.zeros_(attn1.W_o.weight)

    attn1.W_q.weight.data[0, IS_LAST_DIM] = 100.0
    attn1.W_k.weight.data[0, IS_HYP_DIM]  = 1.0

    # V output dim d (0..27)   = HYP_LEN * char_one_hot[d]
    # V output dim 28+k         = HYP_LEN * bigram_det[k]
    for d in range(28):
        attn1.W_v.weight.data[d, d] = float(HYP_LEN)
    for k in range(K):
        attn1.W_v.weight.data[28 + k, BIG_BASE + k] = float(HYP_LEN)
    for d in range(28):
        attn1.W_o.weight.data[HCOUNT_BASE + d, d] = 1.0
    for k in range(K):
        attn1.W_o.weight.data[BCOUNT_BASE + k, 28 + k] = 1.0

    # Layer 1 MLP: zero (passthrough)
    mlp1 = model.blocks[1].mlp
    nn.init.zeros_(mlp1.fc1.weight); nn.init.zeros_(mlp1.fc1.bias)
    nn.init.zeros_(mlp1.fc2.weight); nn.init.zeros_(mlp1.fc2.bias)

    # =====================================================================
    # Head: hand-coded multinomial LR over (char counts, bigram counts)
    # =====================================================================
    SCALE = 10.0
    nn.init.zeros_(model.head.weight)
    for k, label_id in enumerate([id_0, id_1, id_2]):
        for d in range(28):
            model.head.weight.data[label_id, HCOUNT_BASE + d] = SCALE * CHAR_COEF[k][d]
        for b in range(K):
            model.head.weight.data[label_id, BCOUNT_BASE + b] = SCALE * BIGRAM_COEF[k][b]
        # Intercept routed via POS_CONST_DIM (=1 at every position incl. LAST)
        model.head.weight.data[label_id, POS_CONST_DIM] = SCALE * INTERCEPT[k]


model_shorthand_name = "BigramDetector150"
model_description = "Iter22: bigram-detection circuit with K=150 hypothesis bigrams (vs iter21\'s 60). Same architecture - L0 shift attn + bigram MLP, L1 uniform-over-hyp counter, head=multinomial-LR. d_model=400, 1 head, 2 layers, d_ff=150."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def upsert_overall_results(rows: list[dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("task")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])
    print(f"Overall results saved → {path}")



def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(
        vocab_size=task.vocab_size, max_seq_len=max_seq_len,
        d_model=400, n_heads=1, n_layers=2, d_ff=150,
    )
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name (see src/task.py)", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "",
        "model_shorthand_name":  model_shorthand_name,
        "n_params":    f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
