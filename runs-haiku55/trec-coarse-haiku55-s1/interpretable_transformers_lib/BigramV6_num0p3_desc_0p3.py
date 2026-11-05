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
# Task-specific design constants (trec-coarse)
# ---------------------------------------------------------------------------
# Classes (answer digit = 30 + class id):
ABBR, ENTY, DESC, HUM, LOC, NUM = range(6)
CLASS_NAMES = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]

# Word-level vote table: (word, class, weight). A word is "present" when it is
# surrounded by boundaries (space, '_' padding, or '=').
_RAW_VOTES = [
    # HUM
    ("who", HUM, 4.0), ("whom", HUM, 3.0), ("whose", HUM, 3.0),
    ("person", HUM, 2.0), ("people", HUM, 2.0), ("man", HUM, 1.5), ("woman", HUM, 1.5),
    ("president", HUM, 1.5), ("author", HUM, 1.5), ("actor", HUM, 1.5), ("actress", HUM, 1.5),
    ("founder", HUM, 1.5), ("inventor", HUM, 1.5), ("king", HUM, 1.5), ("queen", HUM, 1.5),
    ("singer", HUM, 1.5), ("writer", HUM, 1.5), ("leader", HUM, 1.5), ("group", HUM, 1.0),
    ("athlete", HUM, 1.5), ("crooner", HUM, 1.5), ("composer", HUM, 1.5), ("artist", HUM, 1.5),
    ("painter", HUM, 1.5), ("scientist", HUM, 1.5), ("doctor", HUM, 1.5), ("explorer", HUM, 1.5),
    ("general", HUM, 1.5), ("emperor", HUM, 1.5), ("pope", HUM, 1.5), ("ruler", HUM, 1.5),
    ("father", HUM, 1.5), ("mother", HUM, 1.5), ("husband", HUM, 1.5), ("wife", HUM, 1.5),
    ("teacher", HUM, 1.5), ("lawyer", HUM, 1.5), ("director", HUM, 1.5), ("poet", HUM, 1.5),
    ("army", HUM, 1.5), ("company", HUM, 1.0), ("team", HUM, 1.0),
    # LOC
    ("where", LOC, 4.0), ("country", LOC, 2.5), ("countries", LOC, 2.5), ("city", LOC, 2.5),
    ("cities", LOC, 2.5), ("state", LOC, 2.0), ("states", LOC, 2.0), ("river", LOC, 2.5),
    ("mountain", LOC, 2.5), ("island", LOC, 2.5), ("capital", LOC, 2.0), ("located", LOC, 1.5),
    ("location", LOC, 2.0), ("town", LOC, 2.0), ("continent", LOC, 2.5), ("lake", LOC, 2.0),
    ("ocean", LOC, 2.0), ("sea", LOC, 2.0), ("province", LOC, 2.0), ("county", LOC, 2.5),
    ("place", LOC, 1.5), ("region", LOC, 2.0), ("nation", LOC, 2.0), ("park", LOC, 1.5),
    ("street", LOC, 2.0),
    ("mountains", LOC, 2.5), ("rivers", LOC, 2.5), ("islands", LOC, 2.5), ("lakes", LOC, 2.0),
    ("oceans", LOC, 2.0), ("seas", LOC, 2.0), ("regions", LOC, 2.0), ("towns", LOC, 2.0),
    # NUM
    ("when", NUM, 4.0), ("year", NUM, 2.0), ("years", NUM, 2.0), ("date", NUM, 2.5),
    ("dates", NUM, 2.5), ("numbers", NUM, 2.5), ("amount", NUM, 1.5), ("rate", NUM, 1.5),
    ("average", NUM, 1.0), ("pounds", NUM, 2.0), ("inches", NUM, 2.0), ("degrees", NUM, 2.0),
    ("century", NUM, 2.5), ("month", NUM, 2.5), ("day", NUM, 1.5), ("many", NUM, 3.0),
    ("much", NUM, 2.5), ("long", NUM, 2.0), ("far", NUM, 2.5), ("tall", NUM, 2.5),
    ("high", NUM, 2.0), ("old", NUM, 2.5), ("number", NUM, 2.5), ("percent", NUM, 2.5),
    ("cost", NUM, 2.5), ("price", NUM, 2.5), ("distance", NUM, 2.5), ("weight", NUM, 2.0),
    ("height", NUM, 2.0), ("length", NUM, 2.0), ("speed", NUM, 2.0), ("age", NUM, 2.5),
    ("dollars", NUM, 2.5), ("money", NUM, 2.0), ("miles", NUM, 2.0), ("feet", NUM, 2.0),
    ("birthday", NUM, 2.0), ("time", NUM, 1.0), ("minutes", NUM, 2.0), ("hours", NUM, 2.0),
    # DESC
    ("why", DESC, 4.0), ("reason", DESC, 2.5), ("cause", DESC, 2.5), ("mean", DESC, 2.5),
    ("means", DESC, 2.0), ("meaning", DESC, 2.0), ("define", DESC, 2.5), ("describe", DESC, 2.0),
    ("is", DESC, 0.5), ("are", DESC, 0.5), ("how", DESC, 0.5), ("does", DESC, 1.0),
    ("do", DESC, 1.0), ("did", DESC, 1.0), ("was", DESC, 0.5), ("causes", DESC, 2.5),
    # ENTY
    ("what", ENTY, 0.4), ("which", ENTY, 0.4), ("animal", ENTY, 2.5), ("animals", ENTY, 2.5),
    ("plants", ENTY, 2.5), ("foods", ENTY, 2.5), ("colors", ENTY, 2.5), ("languages", ENTY, 2.5),
    ("games", ENTY, 2.0), ("songs", ENTY, 2.0), ("films", ENTY, 1.5), ("movies", ENTY, 1.5),
    ("books", ENTY, 2.0), ("cars", ENTY, 2.0), ("birds", ENTY, 2.5), ("dogs", ENTY, 2.5),
    ("diseases", ENTY, 2.5), ("drinks", ENTY, 2.0), ("metals", ENTY, 2.5), ("trees", ENTY, 2.5),
    ("flowers", ENTY, 2.5),
    ("color", ENTY, 2.5), ("colour", ENTY, 2.5), ("sport", ENTY, 2.5), ("sports", ENTY, 2.5),
    ("food", ENTY, 2.5), ("language", ENTY, 2.5), ("plant", ENTY, 2.5), ("flower", ENTY, 2.5),
    ("game", ENTY, 2.0), ("music", ENTY, 2.0), ("song", ENTY, 2.0), ("film", ENTY, 2.0),
    ("movie", ENTY, 1.5), ("book", ENTY, 2.0), ("car", ENTY, 2.0), ("drink", ENTY, 2.0),
    ("disease", ENTY, 2.5), ("word", ENTY, 2.0), ("symbol", ENTY, 2.5), ("religion", ENTY, 2.5),
    ("currency", ENTY, 2.5), ("bird", ENTY, 2.5), ("fish", ENTY, 2.5), ("dog", ENTY, 2.5),
    ("element", ENTY, 2.0), ("substance", ENTY, 2.0), ("product", ENTY, 2.0), ("vehicle", ENTY, 2.0),
    ("invention", ENTY, 2.0), ("material", ENTY, 2.0), ("metal", ENTY, 2.5), ("tree", ENTY, 2.5),
    ("insect", ENTY, 2.5),
    # ABBR
    ("acronym", ABBR, 3.0), ("stand", ABBR, 2.0), ("full", ABBR, 1.5), ("initials", ABBR, 2.0),
    # --- vocabulary expansion (V5) ---
    # HUM people / roles
    ("engineer", HUM, 1.5), ("pilot", HUM, 1.5), ("soldier", HUM, 1.5), ("priest", HUM, 1.5),
    ("nurse", HUM, 1.5), ("student", HUM, 1.5), ("child", HUM, 1.0), ("children", HUM, 1.0),
    ("boy", HUM, 1.5), ("girl", HUM, 1.5), ("celebrity", HUM, 1.5), ("hero", HUM, 1.5),
    ("villain", HUM, 1.5), ("friend", HUM, 1.5), ("brother", HUM, 1.5), ("sister", HUM, 1.5),
    ("daughter", HUM, 1.5), ("son", HUM, 1.0), ("member", HUM, 1.0), ("members", HUM, 1.0),
    ("boss", HUM, 1.0), ("partner", HUM, 1.0),
    # LOC places
    ("road", LOC, 1.5), ("bridge", LOC, 2.0), ("airport", LOC, 2.5), ("hospital", LOC, 1.5),
    ("museum", LOC, 1.5), ("stadium", LOC, 1.5), ("palace", LOC, 1.5), ("castle", LOC, 1.5),
    ("valley", LOC, 2.0), ("desert", LOC, 2.0), ("forest", LOC, 2.0), ("beach", LOC, 2.0),
    ("coast", LOC, 2.0), ("border", LOC, 1.5), ("village", LOC, 2.0), ("born", NUM, 1.5),
    # NUM quantities / units / time
    ("birth", NUM, 1.5), ("salary", NUM, 2.0), ("fee", NUM, 1.5), ("ratio", NUM, 2.0),
    ("count", NUM, 2.0), ("total", NUM, 1.5), ("quantity", NUM, 2.0), ("volume", NUM, 2.0),
    ("width", NUM, 2.0), ("depth", NUM, 2.0), ("week", NUM, 2.0), ("weeks", NUM, 2.0),
    ("decade", NUM, 2.5), ("decades", NUM, 2.5), ("seconds", NUM, 2.0), ("hour", NUM, 1.5),
    ("minute", NUM, 1.5), ("ounces", NUM, 2.0), ("gallons", NUM, 2.0), ("mph", NUM, 2.0),
    ("pound", NUM, 2.0), ("dollar", NUM, 2.5), ("cents", NUM, 2.5), ("aged", NUM, 2.0),
    # DESC processes / explanations
    ("purpose", DESC, 1.5), ("method", DESC, 1.5), ("process", DESC, 1.5), ("explain", DESC, 2.0),
    ("example", DESC, 1.5), ("examples", DESC, 1.5), ("steps", DESC, 1.5), ("effect", DESC, 2.0),
    ("effects", DESC, 2.0), ("origin", DESC, 2.5), ("origins", DESC, 2.5), ("symptoms", DESC, 2.0),
    ("symptom", DESC, 2.0), ("treatment", DESC, 2.0), ("prevent", DESC, 2.0), ("cure", DESC, 2.0),
    ("work", DESC, 1.5), ("works", DESC, 1.5), ("function", DESC, 1.5), ("role", DESC, 1.5),
    # ENTY objects / animals / food / arts
    ("tool", ENTY, 2.0), ("weapon", ENTY, 2.5), ("machine", ENTY, 2.0), ("drug", ENTY, 2.5),
    ("medicine", ENTY, 2.5), ("chemical", ENTY, 2.5), ("chemicals", ENTY, 2.5), ("gas", ENTY, 2.0),
    ("snake", ENTY, 2.5), ("horse", ENTY, 2.5), ("horses", ENTY, 2.5), ("cat", ENTY, 2.5),
    ("cats", ENTY, 2.5), ("cow", ENTY, 2.5), ("lion", ENTY, 2.5), ("bear", ENTY, 2.5),
    ("whale", ENTY, 2.5), ("shark", ENTY, 2.5), ("monkey", ENTY, 2.5), ("fruit", ENTY, 2.5),
    ("fruits", ENTY, 2.5), ("vegetable", ENTY, 2.5), ("candy", ENTY, 2.0), ("wine", ENTY, 2.0),
    ("beer", ENTY, 2.0), ("coffee", ENTY, 2.0), ("bread", ENTY, 2.0), ("cheese", ENTY, 2.0),
    ("show", ENTY, 1.5), ("shows", ENTY, 1.5), ("program", ENTY, 1.5), ("newspaper", ENTY, 2.0),
    ("magazine", ENTY, 2.0), ("opera", ENTY, 2.5), ("painting", ENTY, 2.0), ("novel", ENTY, 2.0),
    ("poem", ENTY, 2.0), ("album", ENTY, 2.5), ("cartoon", ENTY, 2.5), ("cartoons", ENTY, 2.5),
    ("holiday", ENTY, 2.0), ("festival", ENTY, 2.0), ("war", ENTY, 1.5), ("battle", ENTY, 2.0),
    ("event", ENTY, 2.0), ("race", ENTY, 2.0), ("award", ENTY, 1.5), ("alphabet", ENTY, 2.5),
    ("dance", ENTY, 2.5), ("guitar", ENTY, 2.5), ("piano", ENTY, 2.5), ("violin", ENTY, 2.5),
    ("drum", ENTY, 2.5), ("ball", ENTY, 2.0), ("coin", ENTY, 2.5), ("gold", ENTY, 2.0),
    ("diamond", ENTY, 2.5), ("silver", ENTY, 2.0), ("iron", ENTY, 2.0), ("oil", ENTY, 2.0),
    ("plastic", ENTY, 2.0), ("glass", ENTY, 2.0), ("fuel", ENTY, 2.0),
    # --- vocabulary expansion batch 2 (V6) ---
    # HUM
    ("chef", HUM, 1.5), ("farmer", HUM, 1.5), ("monarch", HUM, 1.5), ("prince", HUM, 1.5),
    ("princess", HUM, 1.5), ("saint", HUM, 1.5), ("god", HUM, 1.5), ("goddess", HUM, 1.5),
    ("rapper", HUM, 1.5), ("dancer", HUM, 1.5), ("player", HUM, 1.5), ("players", HUM, 1.5),
    ("coach", HUM, 1.5), ("manager", HUM, 1.5), ("owner", HUM, 1.5), ("employee", HUM, 1.5),
    ("worker", HUM, 1.5), ("workers", HUM, 1.5), ("citizen", HUM, 1.5), ("citizens", HUM, 1.5),
    ("resident", HUM, 1.5), ("candidate", HUM, 1.5), ("hunter", HUM, 1.5), ("thief", HUM, 1.5),
    ("killer", HUM, 1.5), ("murderer", HUM, 1.5), ("criminal", HUM, 1.5), ("spy", HUM, 1.5),
    ("fan", HUM, 1.5), ("fans", HUM, 1.5), ("audience", HUM, 1.0), ("crowd", HUM, 1.0),
    ("family", HUM, 1.5), ("families", HUM, 1.5), ("baby", HUM, 1.5), ("kid", HUM, 1.0),
    ("parent", HUM, 1.5), ("parents", HUM, 1.5), ("uncle", HUM, 1.5), ("aunt", HUM, 1.5),
    ("cousin", HUM, 1.5), ("wizard", HUM, 1.5), ("knight", HUM, 1.5), ("warrior", HUM, 1.5),
    ("monk", HUM, 1.5), ("nun", HUM, 1.5), ("bishop", HUM, 1.5), ("governor", HUM, 1.5),
    ("mayor", HUM, 1.5), ("senator", HUM, 1.5), ("minister", HUM, 1.5), ("judge", HUM, 1.5),
    ("officer", HUM, 1.5), ("reporter", HUM, 1.5), ("editor", HUM, 1.5), ("publisher", HUM, 1.5),
    ("inventors", HUM, 1.5), ("founders", HUM, 1.5),
    # LOC
    ("capitals", LOC, 2.5), ("nations", LOC, 2.0), ("provinces", LOC, 2.0), ("cape", LOC, 1.5),
    ("bay", LOC, 1.5), ("gulf", LOC, 1.5), ("harbor", LOC, 2.0), ("port", LOC, 1.5),
    ("address", LOC, 2.0), ("land", LOC, 1.5), ("area", LOC, 1.5), ("zone", LOC, 1.5),
    ("building", LOC, 1.5), ("tower", LOC, 1.5), ("monument", LOC, 2.0), ("landmark", LOC, 2.0),
    ("landmarks", LOC, 2.0), ("hill", LOC, 2.0), ("hills", LOC, 2.0), ("volcano", LOC, 2.5),
    ("canyon", LOC, 2.0), ("cave", LOC, 1.5), ("house", LOC, 1.0), ("home", LOC, 1.0),
    ("north", LOC, 1.5), ("south", LOC, 1.5), ("east", LOC, 1.5), ("west", LOC, 1.5),
    ("northern", LOC, 1.5), ("southern", LOC, 1.5), ("eastern", LOC, 1.5), ("western", LOC, 1.5),
    # NUM
    ("ages", NUM, 2.5), ("score", NUM, 2.0), ("scores", NUM, 2.0), ("points", NUM, 1.5),
    ("mass", NUM, 2.0), ("calories", NUM, 2.5), ("tons", NUM, 2.5), ("ton", NUM, 2.5),
    ("kilos", NUM, 2.5), ("meters", NUM, 2.5), ("yards", NUM, 2.5), ("acres", NUM, 2.5),
    ("square", NUM, 1.5), ("per", NUM, 1.5), ("rates", NUM, 1.5), ("fraction", NUM, 2.0),
    ("days", NUM, 2.0), ("months", NUM, 2.5), ("july", NUM, 2.5), ("august", NUM, 2.5),
    ("december", NUM, 2.5), ("birthdays", NUM, 2.0), ("era", NUM, 2.0), ("period", NUM, 1.5),
    # DESC
    ("reasons", DESC, 2.5), ("results", DESC, 1.5), ("result", DESC, 1.5), ("solution", DESC, 2.0),
    ("answer", DESC, 1.5), ("answers", DESC, 1.5), ("concept", DESC, 2.0), ("idea", DESC, 1.5),
    ("theory", DESC, 2.0), ("theories", DESC, 2.0), ("principle", DESC, 2.0), ("approach", DESC, 1.5),
    ("technique", DESC, 1.5), ("mechanism", DESC, 2.0), ("system", DESC, 1.5), ("structure", DESC, 1.5),
    ("property", DESC, 1.5), ("formula", DESC, 2.5), ("recipe", DESC, 2.5), ("guide", DESC, 1.5),
    ("tips", DESC, 1.5), ("advice", DESC, 2.0), ("benefits", DESC, 1.5), ("dangers", DESC, 1.5),
    ("danger", DESC, 1.5), ("risks", DESC, 1.5), ("risk", DESC, 1.5), ("problem", DESC, 1.5),
    ("problems", DESC, 1.5), ("source", DESC, 1.5), ("sources", DESC, 1.5), ("history", DESC, 1.5),
    ("tradition", DESC, 1.5), ("custom", DESC, 1.5), ("rules", DESC, 1.5), ("rule", DESC, 1.5),
    ("law", DESC, 1.0), ("laws", DESC, 1.0),
    # ENTY
    ("fabric", ENTY, 2.0), ("clothing", ENTY, 2.0), ("clothes", ENTY, 2.0), ("shoes", ENTY, 2.0),
    ("hat", ENTY, 2.0), ("dress", ENTY, 2.0), ("shape", ENTY, 2.0), ("shapes", ENTY, 2.0),
    ("letter", ENTY, 2.0), ("letters", ENTY, 2.0), ("sound", ENTY, 1.5), ("sounds", ENTY, 1.5),
    ("bomb", ENTY, 2.0), ("bombs", ENTY, 2.0), ("missile", ENTY, 2.0), ("rocket", ENTY, 2.0),
    ("planet", ENTY, 1.5), ("planets", ENTY, 1.5), ("moon", ENTY, 1.5), ("sun", ENTY, 1.5),
    ("galaxy", ENTY, 2.0), ("mineral", ENTY, 2.0), ("rock", ENTY, 2.0), ("rocks", ENTY, 2.0),
    ("mammal", ENTY, 2.5), ("reptile", ENTY, 2.5), ("species", ENTY, 2.5), ("breed", ENTY, 2.5),
    ("breeds", ENTY, 2.5), ("virus", ENTY, 2.5), ("bacteria", ENTY, 2.5), ("vitamin", ENTY, 2.5),
    ("vitamins", ENTY, 2.5), ("sugar", ENTY, 2.0), ("salt", ENTY, 2.0), ("meat", ENTY, 2.0),
    ("dish", ENTY, 2.0), ("dishes", ENTY, 2.0), ("cocktail", ENTY, 2.0), ("soup", ENTY, 2.0),
    ("cake", ENTY, 2.0), ("pizza", ENTY, 2.0), ("sandwich", ENTY, 2.0), ("meal", ENTY, 2.0),
    ("tune", ENTY, 2.0), ("tunes", ENTY, 2.0), ("lyrics", ENTY, 2.0), ("melody", ENTY, 2.0),
    ("genre", ENTY, 2.0), ("theater", ENTY, 2.0), ("theatre", ENTY, 2.0), ("radio", ENTY, 2.0),
    ("computer", ENTY, 2.0), ("computers", ENTY, 2.0), ("software", ENTY, 2.0), ("internet", ENTY, 2.0),
    ("phone", ENTY, 2.0), ("phones", ENTY, 2.0), ("camera", ENTY, 2.0), ("engine", ENTY, 2.0),
    ("engines", ENTY, 2.0), ("ship", ENTY, 2.0), ("ships", ENTY, 2.0), ("boat", ENTY, 2.0),
    ("airplane", ENTY, 2.0), ("plane", ENTY, 2.0), ("train", ENTY, 2.0), ("bus", ENTY, 2.0),
    ("bike", ENTY, 2.0), ("truck", ENTY, 2.0), ("sword", ENTY, 2.0), ("shield", ENTY, 2.0),
    ("armor", ENTY, 2.0), ("gun", ENTY, 2.0),
]

# Two-word detectors: (first word, second word, class, weight). Matches the phrase
# "w1 w2" with boundaries around it.  Used to disambiguate e.g. "how long" vs "how is".
_BI_VOTES = [
    ("how", "many", NUM, 3.5), ("how", "much", NUM, 3.0), ("how", "long", NUM, 2.5),
    ("how", "old", NUM, 3.0), ("how", "far", NUM, 3.0), ("how", "tall", NUM, 3.0),
    ("how", "high", NUM, 2.5), ("how", "fast", NUM, 2.5), ("how", "large", NUM, 2.5),
    ("how", "deep", NUM, 2.5), ("how", "wide", NUM, 2.5),
    ("how", "did", DESC, 1.5), ("how", "do", DESC, 1.5), ("how", "does", DESC, 1.5),
    ("how", "can", DESC, 1.5),
    ("what", "is", DESC, 2.0), ("what", "are", DESC, 1.5),
    ("what", "kind", ENTY, 2.0), ("what", "type", ENTY, 2.0), ("what", "sort", ENTY, 2.0),
    ("who", "was", HUM, 1.5), ("who", "is", HUM, 1.5), ("who", "were", HUM, 1.5),
]

# Per-class bias added to the logits. Only NUM is nudged (+0.3), so a question that
# no vote covers is labelled NUM instead of ABBR.
CLASS_BIAS = [0.0, 0.0, 0.3, 0.0, 0.0, 0.3]

UNI_WORDS: list[str] = sorted({w for w, _, _ in _RAW_VOTES})
N_UNI = len(UNI_WORDS)
N_BI = len(_BI_VOTES)
N_DET = 2 * N_UNI + 2 * N_BI  # detectors: [unigrams | question-initial unigrams | bigrams | question-initial bigrams]
N_CHAR = 36                 # task vocabulary (letters, space, ', _, =, digits)
BOUNDARY_CHARS = [" ", "_", "="]
POS_MAX = 82                # max_seq_len used by the task (81 prompt tokens + answer)
N_OFF = 13                  # relative offsets 0..12 copied in layer 1 (bigrams up to ~12 chars)
START_BONUS = 0.5           # question-initial unigram votes get (1 + START_BONUS) weight
UNI_SCALE = 1.0             # global multiplier on unigram vote weights
BI_SCALE = 1.0              # global multiplier on phrase vote weights

# Residual-stream layout
TOK0 = 0                    # [0, 36): token one-hot
CONST = N_CHAR              # 36: constant 1 at every position (used for biases)
POS0 = CONST + 1            # [37, 37+82): position one-hot
F1_0 = POS0 + POS_MAX       # [119, ...): N_OFF blocks of 36 (block k = token at offset k)
F2_0 = F1_0 + N_OFF * N_CHAR  # detector indicator at each position (N_DET dims)
F3_0 = F2_0 + N_DET         # pooled detector counts at the final position (N_DET dims)
D_MODEL = F3_0 + N_DET


def _det_index_uni(u: int) -> int:
    return u


def _det_index_start(u: int) -> int:
    return N_UNI + u


def _det_index_bi(b: int) -> int:
    return 2 * N_UNI + b


def _det_index_bis(b: int) -> int:
    return 2 * N_UNI + N_BI + b


def _detector_terms() -> list[tuple[list[tuple[int, str]], int | None]]:
    """For each detector, return (terms, pos_term).

    terms: list of (offset, token) pairs that must all hold at the current position p
           (token at offset k is the char at p-k). pos_term: if not None, position p must
           equal this value (used for the question-initial variant).
    """
    dets = []
    # base unigram: [boundary] w [boundary] with a real boundary before the word
    for w in UNI_WORDS:
        L = len(w)
        terms = []
        for k in range(1, L + 1):
            terms.append((k, w[L - k]))
        dets.append((("bd0", terms, L), None))
    # question-initial unigram: same letters, word starts at position 0 (pos == L)
    for w in UNI_WORDS:
        dets.append((("start", [(k, w[len(w) - k]) for k in range(1, len(w) + 1)], len(w)), len(w)))
    # bigram detectors
    for (w1, w2, _, _) in _BI_VOTES:
        L1, L2 = len(w1), len(w2)
        assert L1 + L2 + 2 < N_OFF, (w1, w2)
        terms = []
        for k in range(1, L2 + 1):          # second word letters
            terms.append((k, w2[L2 - k]))
        # between boundary at offset L2+1 is handled as a boundary-set term below
        for k in range(L2 + 2, L2 + 2 + L1):  # first word letters
            terms.append((k, w1[L1 - (k - (L2 + 2)) - 1]))
        dets.append((("bi", terms, L1 + L2), None))
    # question-initial bigrams: the first word starts at position 0, so the phrase ends at p = L1+L2+1
    for (w1, w2, _, _) in _BI_VOTES:
        L1, L2 = len(w1), len(w2)
        terms = []
        for k in range(1, L2 + 1):
            terms.append((k, w2[L2 - k]))
        for k in range(L2 + 2, L2 + 2 + L1):
            terms.append((k, w1[L1 - (k - (L2 + 2)) - 1]))
        dets.append((("bis", terms, L1 + L2), L1 + L2 + 1))
    return dets


DET_SPECS = _detector_terms()
assert len(DET_SPECS) == N_DET


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm-free block: LayerNorm is removed so features stay exactly linear."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads, d_head)
        self.ln2 = nn.Identity()
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """2-layer causal transformer (no LayerNorm). Layer 1: relative-offset heads +
    word/phrase-detector MLP. Layer 2: uniform pooling over the question."""

    def __init__(self, vocab_size: int, max_seq_len: int = 82):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = D_MODEL

        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList([
            Block(D_MODEL, N_OFF, POS_MAX, d_ff=N_DET),   # offsets + detectors
            Block(D_MODEL, 1, N_DET, d_ff=1),              # uniform pooling
        ])
        self.final_ln = nn.Identity()
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

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
    """Populate `model`'s parameters in-place. No training allowed."""
    torch.manual_seed(0)
    stoi = task.stoi
    assert model.vocab_size == N_CHAR
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # --- embeddings ------------------------------------------------------
        for ch, t in stoi.items():
            model.token_emb.weight[t, TOK0 + t] = 1.0
        model.token_emb.weight[:, CONST] = 1.0
        for p_ in range(model.max_seq_len):
            model.pos_emb.weight[p_, POS0 + p_] = 1.0

        # --- layer 1 attention: head k copies the token at offset k ----------
        attn1 = model.blocks[0].attn
        c = math.sqrt(30.0 * math.sqrt(POS_MAX))  # q.k/sqrt(dh) = 30 on a match
        for k in range(N_OFF):
            base = k * POS_MAX
            for i in range(POS_MAX):
                if i + k < model.max_seq_len:
                    attn1.W_q.weight[base + i, POS0 + i + k] = c   # query for pos p=i+k wants key i=p-k
                attn1.W_k.weight[base + i, POS0 + i] = c           # key at pos j
            for d in range(N_CHAR):
                attn1.W_v.weight[base + d, TOK0 + d] = 1.0
                attn1.W_o.weight[F1_0 + N_CHAR * k + d, base + d] = 1.0

        # --- layer 1 MLP: detectors (exact AND via 2s - 2n + 1 threshold) ----
        mlp1 = model.blocks[0].mlp
        bnd_ids = [stoi[b] for b in BOUNDARY_CHARS]
        for det_i, ((kind, terms, n_letters), pos_term) in enumerate(DET_SPECS):
            # terms: letter constraints; add boundary constraints depending on kind
            if kind == "bd0":           # base unigram: trailing boundary + leading boundary
                L = n_letters
                cols = [[(0, b) for b in bnd_ids], [(L + 1, b) for b in bnd_ids]]
                n_terms = L + 2
            elif kind == "start":       # question-initial unigram: trailing boundary + POS
                L = n_letters
                cols = [[(0, b) for b in bnd_ids]]
                n_terms = L + 2
            else:                       # bigram (base or question-initial): trailing + between + leading
                n_terms = len(terms) + 3
            u = det_i

            if kind == "bd0":
                for k, tok in terms:
                    mlp1.fc1.weight[u, F1_0 + N_CHAR * k + stoi[tok]] = 2.0
                for o, b in cols[0] + cols[1]:      # b is already a token id
                    mlp1.fc1.weight[u, F1_0 + N_CHAR * o + b] = 2.0
            elif kind == "start":
                for k, tok in terms:
                    mlp1.fc1.weight[u, F1_0 + N_CHAR * k + stoi[tok]] = 2.0
                for o, b in cols[0]:
                    mlp1.fc1.weight[u, F1_0 + N_CHAR * o + b] = 2.0
                mlp1.fc1.weight[u, POS0 + n_letters] = 2.0
            else:                       # bigram; kind "bi" has a leading boundary, "bis" has pos term
                bi_idx = det_i - 2 * N_UNI - (N_BI if kind == "bis" else 0)
                w1, w2, _, _ = _BI_VOTES[bi_idx]
                L1, L2 = len(w1), len(w2)
                for k, tok in terms:
                    mlp1.fc1.weight[u, F1_0 + N_CHAR * k + stoi[tok]] = 2.0
                for b in bnd_ids:                                # trailing boundary
                    mlp1.fc1.weight[u, F1_0 + b] = 2.0
                for b in bnd_ids:                                # boundary between the words
                    mlp1.fc1.weight[u, F1_0 + N_CHAR * (L2 + 1) + b] = 2.0
                if kind == "bi":
                    for b in bnd_ids:                            # leading boundary before w1
                        mlp1.fc1.weight[u, F1_0 + N_CHAR * (L1 + L2 + 2) + b] = 2.0
                else:
                    mlp1.fc1.weight[u, POS0 + L1 + L2 + 1] = 2.0   # w1 starts the question
            mlp1.fc1.bias[u] = -2.0 * n_terms + 1.0
            mlp1.fc2.weight[F2_0 + u, u] = 1.0

        # --- layer 2: uniform attention over all positions = pooled counts ---
        attn2 = model.blocks[1].attn
        for d in range(N_DET):
            attn2.W_v.weight[d, F2_0 + d] = 1.0      # value = detector indicator
            attn2.W_o.weight[F3_0 + d, d] = float(model.max_seq_len - 1)  # undo the 1/81 average

        # --- output head: detector votes -> class logits on answer tokens ----
        head = model.head.weight
        answer_ids = [stoi[str(c)] for c in range(6)]
        for v in range(model.vocab_size):
            if v not in answer_ids:
                head[v, CONST] = -1000.0   # never predict a non-answer token
        for c, aid in enumerate(answer_ids):
            head[aid, CONST] = CLASS_BIAS[c]
        for word, cls, weight in _RAW_VOTES:
            u = UNI_WORDS.index(word)
            w = weight * UNI_SCALE
            head[answer_ids[cls], F3_0 + _det_index_uni(u)] += w
            head[answer_ids[cls], F3_0 + _det_index_start(u)] += (1.0 + START_BONUS) * w
        for b, (w1, w2, cls, weight) in enumerate(_BI_VOTES):
            w = weight * BI_SCALE
            head[answer_ids[cls], F3_0 + _det_index_bi(b)] += w
            head[answer_ids[cls], F3_0 + _det_index_bis(b)] += (1.0 + START_BONUS) * w


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "BigramV6_num0p3_desc_0p3"
model_description = (
    "BigramV6_num_bias_0p3 plus a +0.3 DESC class bias."
)


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
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
