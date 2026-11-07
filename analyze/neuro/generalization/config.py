"""Selection of best non-redundant models per run + the experiment matrix.

Train/test split matches the original runs (num_train=8, the 3 standard test
stories). New-subject runs reuse the identical stimuli on UTS01/UTS02. New-story
runs hold the same subject (UTS03) but evaluate on stories #8-10 of the shared
training pool (never used to fit any of these models).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # for the `src` symlink → experiments/evolve-neuro/src
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUNS_DIR = os.path.join(REPO, "experiments", "runs", "neuro")

# The full shared story sets (intersection of UTS01/UTS02/UTS03 huge training sets).
from src import data  # noqa: E402

TRAIN_STORIES = data.TRAIN_STORIES        # 93 shared stories
TEST_STORIES = data.TEST_STORIES          # 3 shared held-out test stories

ORIG_TRAIN = TRAIN_STORIES[:8]            # the 8 stories every model was fit on
ORIG_TEST = TEST_STORIES[:3]              # original held-out test stories
# New-story transfer set: ALL shared stories not used for training (first 8) or testing.
# (TEST_STORIES are a separate list, so the held-out pool is simply TRAIN_STORIES[8:].)
NEW_STORIES = TRAIN_STORIES[8:]           # 85 held-out stories
NEW_SUBJECTS = ["UTS01", "UTS02"]
ORIG_SUBJECT = "UTS03"

# (run_folder, model_name, reported_test_corr, note)
# ~2-3 best non-redundant models per run (distinct method families where possible).
SELECTED = [
    # Jun03 run1 — Claude Opus 4.8 (medium). LexFeat interpretable-feature-token transformer.
    ("fmri-jun3-run1", "LexFeatFreqMerge", 0.0745, "best LexFeat (full feature set)"),
    ("fmri-jun3-run1", "LexFeatBoC", 0.0503, "ablated LexFeat bag-of-categories"),
    # Jun03 run2 — GPT-5.5 (xhigh). Greedy semantic best-lexicon search + structural variants.
    ("fmri-jun03-run2", "semantic_bestlex_compact_v94", 0.0585, "compact best-lexicon"),
    ("fmri-jun03-run2", "lexsem_rec70_tail24_v09", 0.0514, "lexical+semantic recency"),
    ("fmri-jun03-run2", "content_structure_v13", 0.0458, "content+structure multidecay"),
    # Jun03 run3 — Gemini 3.1 Pro (high). Word-boundary / temporal-pool / deep-ensemble circuits.
    ("fmri-jun03-run3", "WordBoundaryFeatures", 0.0405, "word-boundary features"),
    ("fmri-jun03-run3", "Deep_EnsembleWB_Tuned", 0.0370, "tuned deep ensemble"),
    ("fmri-jun03-run3", "MultiScale_Temporal_Pool", 0.0362, "multiscale temporal pooling"),
    # Jun03 run4 — Claude Opus 4.7 (xhigh). FeatBag interpretable feature-bag transformer.
    ("fmri-jun03-run4", "FeatBag_v1116_Emo40_0810_CEILING_BREAK", 0.0810, "near-ceiling FeatBag (emotion-lexicon tuned; run later reached 0.082 with same family)"),
    ("fmri-jun03-run4", "FeatBag_v11_MoreSEM", 0.0702, "earlier best FeatBag (rich semantics)"),
    ("fmri-jun03-run4", "FeatBag_v2_WordID", 0.0550, "ablated FeatBag (word-id)"),
    # May27 run1 — Claude Opus 4.7 (xhigh), UNTRIMMED original eval. WordNet+morphology+perceptual.
    ("fmri-may27-run1", "WordNetMorphLingPerceptual", 0.1146, "best (UNTRIMMED orig eval)"),
    ("fmri-may27-run1", "WordNetMorphLingMultiTau", 0.0665, "simpler multi-tau content"),
    # Jun04 run1 — Claude Opus 4.8 (xhigh). Resumed run4's FeatBag (was given all prior runs' results).
    # NOTE: this run's single highest scores (~0.088) used corpus statistics (n-gram surprisal LM + LSA),
    # which are disallowed; those are excluded here. Its best LEGITIMATE model is the novelty+gazetteer one.
    ("fmri-jun04-run1", "FeatBagNovelty_NamesDense_xrun", 0.0837, "best legitimate: FeatBag + within-story novelty + name gazetteer (beats GPT-2 XL)"),
    ("fmri-jun04-run1", "FeatBag3Head_EmoInt_ConcCat", 0.0792, "3-head FeatBag (emotion-intensity + concreteness)"),
    ("fmri-jun04-run1", "N3_OtherRefBonus8", 0.0774, "earlier FeatBag refinement (other-reference bonus)"),
    ("fmri-jun04-run1", "E24_ContentOnly", 0.0609, "ablated FeatBag (content words only)"),
]


def model_file(run_folder, name):
    return os.path.join(RUNS_DIR, run_folder, "interpretable_transformers_lib", name + ".py")
