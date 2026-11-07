# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build a self-contained interactive HTML report from the story fMRI runs.

Each row of every run's overall_results.csv is one iteration (a hand-written
transformer tried by the coding agent). The metric is `test_corr` on subject
UTS03; GPT-2 XL rows are pretrained baselines. Model / thinking-effort per run
is read from each run folder's metadata.json (recovered from the copilot CLI
logs the first time, then persisted there).
"""
import base64
import csv
import glob
import html
import json
import os
import re

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS_DIR = os.path.join(REPO, "experiments", "runs", "neuro")
OUT = os.path.join(HERE, "report.html")
MAY27_ANALYSIS = os.path.join(RUNS_DIR, "fmri-may27-run1", "analysis")
ASSETS = os.path.join(HERE, "assets")


def img_b64(path):
    """Embed a PNG as a self-contained data URI (keeps report.html portable)."""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

# ---------------------------------------------------------------------------
# Flagging.  The premise is that the agent hand-writes the weights of a small
# interpretable transformer — it may NOT use any model trained on data, INCLUDING
# text pretraining, NO corpus statistics, and the STANDARD eval protocol.  Four
# kinds are disallowed and are excluded from each run's reported best / running-best
# curve:
#   "train"      — fit/back-propped on the fMRI data, OR an oracle that leaks the
#                  held-out responses back into the features.
#   "protocol"   — inflates the score by fitting the (allowed) ridge head on a
#                  non-standard num_train (93 stories vs the fixed pipeline's 8 used
#                  by the GPT-2 XL baseline and every other run) — 11x more training
#                  data, so the number is not comparable. Checked before "corpus"
#                  because these rows disclaim corpus ("no LSA") yet scale num_train.
#   "pretrained" — loads an external pretrained neural encoder (Qwen/BERT/GloVe/…),
#                  i.e. a model trained on data via text pretraining.
#   "corpus"     — derives statistics from the (stimulus) text corpus: an n-gram
#                  surprisal language model, or LSA/PPMI co-occurrence + SVD word
#                  vectors. Explicitly disallowed ("no corpus statistics").
# (The fixed GPT-2 XL baseline is also text-pretrained, but it is the reference
#  point being compared against, not a hand-written entry, so it is not flagged.)
# Detection is keyword-based, then validated against the actual descriptions.
_PRE_RE = re.compile(
    r"qwen|distilbert|deberta|roberta|\bbert\b|glove|word2vec|fasttext|spacy|"
    r"en_core_web|sentence.?transf|llama|mistral|minilm|mpnet|sbert|"
    # run-3's later off-premise push: Gemma-2 / gemma-scope SAE features, the
    # Llama+Qwen+Gemma "SuperEmbedding" PCA stacks, and the Qwen+Mistral
    # "Ensemble_Ultimate" family (some rows only name the family, not the model).
    r"gemma|superembedding|ensemble_ultimate", re.I)
_TRAIN_RE = re.compile(
    r"backprop|end.?to.?end.?train|epochs?\b|\boracle\b|response.?leakage", re.I)
# Non-standard evaluation protocol.  The fixed pipeline trains the ridge head on
# the standard num_train=8 stories (this is what the GPT-2 XL baseline = 0.0826
# and every other run use).  jun-11's last batch instead reports test_corr at
# num_train=93 — 11x more ridge training data — which inflates the number and
# makes it NOT comparable to the baseline or any other run.  The models even
# admit it ("BEATS GPT-2XL 0.0831>0.0826 at num_train=93" but "GPT-2XL also
# scales ... and stays ahead" — at matched data GPT-2 XL hits 0.1348).  Treated
# as training on extra data: flagged and excluded.  Detector validated to match
# exactly those rows and nothing in any other run.
_PROTOCOL_RE = re.compile(
    r"num_train\s*=\s*(?!8\b)\d+|\bntr(?!8\b)\d+|\(8 stories\)\s*->|ntr8\s*->|"
    r"\btraining data\b.{0,15}(lever|biggest|scal|bound)|scales?\s*0?\.0\d+\s*\(8", re.I)
# Corpus-derived distributional / language-model statistics (computed from the
# stimulus text, not hand-coded). The strong tokens below never appear in the
# legitimate hand-built models, so no negation guard is needed.  NOTE: a bare
# "surprisal" is NOT enough — several legitimate hand-coded circuits use the
# word as a hypothesis label (character-delta tracking, fixed English letter-
# frequency constants); only an actual count-LM built over the stimulus corpus
# counts as a violation.
_CORPUS_RE = re.compile(
    r"\bLSA|\bS?PPMI|latent semantic|\bSVD\b|"
    # jun-11's "best stack" is its shorthand for the SPPMI/LSA200/topic-LSA
    # pipeline; a few late variants ("+ max-pool", "+ hashed identity") build on
    # it without re-spelling LSA. The phrase appears in no other run.
    r"best stack|"
    r"surprisal block|n-?gram surprisal|surprisal language model|"
    r"count language model|language model over the stimulus|"
    r"n-?gram language model|distributional.{0,12}semantic", re.I)
# Guards against false positives: "untrained", "FASTTEXT-STYLE" hand-built bags,
# "inspired by", explicitly hand-wired/hand-built circuits.
_NEG_TRAIN_RE = re.compile(r"untrained|no training|not trained|never trained", re.I)
_NEG_PRE_RE = re.compile(r"style|inspired|hand-?built|hand-?wired|hand-?coded", re.I)


def classify(name, desc):
    """Return None | 'train' | 'protocol' | 'corpus' | 'pretrained' for one iteration."""
    blob = f"{name} | {desc}"
    if _TRAIN_RE.search(blob) and not _NEG_TRAIN_RE.search(blob):
        return "train"
    # Checked BEFORE corpus: jun-11's "LEGIT_*" rows disclaim corpus ("no LSA")
    # yet inflate num_train; catching protocol first gives the accurate reason and
    # avoids a false corpus match on the negated "no LSA" text.
    if _PROTOCOL_RE.search(blob):
        return "protocol"
    if _CORPUS_RE.search(blob):
        return "corpus"
    if _PRE_RE.search(blob) and not _NEG_PRE_RE.search(blob):
        return "pretrained"
    return None

# Discover runs from their metadata.json files (written once, then reused).
RUN_META = {}
for mpath in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "metadata.json"))):
    folder = os.path.basename(os.path.dirname(mpath))
    with open(mpath) as f:
        RUN_META[folder] = json.load(f)

# may27 (untrimmed) is shown separately; everything else is directly comparable.
UNTRIMMED = next(f for f, m in RUN_META.items() if m.get("untrimmed"))
# trimmed runs ordered by their label (chronological by construction)
TRIMMED_ORDER = sorted(
    (f for f, m in RUN_META.items() if not m.get("untrimmed")),
    key=lambda f: RUN_META[f]["label"],
)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_run(folder):
    path = os.path.join(RUNS_DIR, folder, "results", "overall_results.csv")
    rows = []
    baseline = None
    it = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            corr = fnum(r["test_corr"])
            if corr is None:
                continue
            name = r.get("model_shorthand_name", "")
            desc = (r.get("description") or "").strip()
            is_base = "gpt" in name.lower()
            rec = {
                "name": name,
                "corr": corr,
                "train": fnum(r.get("train_corr")),
                "params": fnum(r.get("n_params")),
                "desc": desc,
                "base": is_base,
                "flag": None if is_base else classify(name, desc),
            }
            if is_base:
                if baseline is None:
                    baseline = corr
                rec["it"] = None
            else:
                it += 1
                rec["it"] = it
            rows.append(rec)
    # Running max over LEGITIMATE (non-flagged) non-baseline points, in order.
    # Flagged iterations carry the line forward but never raise it.
    best = float("-inf")
    for rec in rows:
        if rec["base"]:
            rec["rmax"] = None
            continue
        if rec["flag"] is None and rec["corr"] > best:
            best = rec["corr"]
        rec["rmax"] = best if best != float("-inf") else None
    nb = [r for r in rows if not r["base"]]
    legit = [r for r in nb if r["flag"] is None]
    flagged = [r for r in nb if r["flag"] is not None]
    best_legit = max(legit, key=lambda r: r["corr"]) if legit else None
    best_any = max(nb, key=lambda r: r["corr"]) if nb else None
    meta = RUN_META[folder]
    return {
        "folder": folder,
        "label": meta["label"],
        "model": meta["model"],
        "effort": meta["effort"],
        "baseline": baseline,
        "n_iter": len(nb),
        "n_train_flag": sum(1 for r in flagged if r["flag"] == "train"),
        "n_pre_flag": sum(1 for r in flagged if r["flag"] == "pretrained"),
        "n_corpus_flag": sum(1 for r in flagged if r["flag"] == "corpus"),
        "n_protocol_flag": sum(1 for r in flagged if r["flag"] == "protocol"),
        # "best" is the best LEGITIMATE (hand-wired, untrained) iteration.
        "best": best_legit["corr"] if best_legit else None,
        "best_name": best_legit["name"] if best_legit else None,
        # best including flagged iterations, reported separately as a caveat.
        "best_any": best_any["corr"] if best_any else None,
        "best_any_name": best_any["name"] if best_any else None,
        "best_any_flag": best_any["flag"] if best_any else None,
        "rows": rows,
    }


DATA = {f: load_run(f) for f in RUN_META}

# ---------------------------------------------------------------------------
# Narrative writeup, grounded in the descriptions / results above.
# ---------------------------------------------------------------------------
WRITEUP = {
    "fmri-jun3-run1": {
        "headline": "Hand-wired interpretable transformers with lexical feature tokens — a strong, compact "
                    "feature-token circuit (just behind run 4's FeatBag ceiling).",
        "worked": "Tokenizing each word into interpretable feature tokens (function-word type, ~40 hand-curated "
                  "semantic categories, perceptual modality, concreteness, animacy, valence/arousal, "
                  "person-reference, morphology) and pooling them with multi-scale recency-weighted attention. "
                  "The <code>LexFeat*</code> family steadily climbed to 0.079 (<code>DisfluencySuppress</code>), with "
                  "small fixes (pronouns, past-tense morphology, spatial prepositions, first-word anchoring, "
                  "frequency de-emphasis, graded repetition-suppression, other-/body-reference and agent-predicate "
                  "boosts, name-pruning and disfluency-suppression) each nudging it up. A late push added "
                  "phonetic rhotic-coda features (<code>PhonRhoticOnly</code> 0.0794) and then numeral/quantity "
                  "tokens — cardinal-number splits, number-repetition boosts and measure-unit words "
                  "(<code>CardinalSplit</code> → <code>MeasureUnit</code>) — finishing at <b>0.0806</b>, "
                  "within 0.002 of the GPT-2 XL baseline.",
        "failed": "Pure bag-of-character and raw semantic-category circuits (<code>RecencyBoC</code> 0.032, "
                  "<code>SemCatBoC</code> 0.034) were far behind — character-level information alone carries "
                  "little fMRI-relevant signal. Adding hand-curated lexical/semantic structure was what mattered.",
    },
    "fmri-jun03-run2": {
        "headline": "GPT-5.5 brute-forced a large lexicon search — many iterations, modest ceiling.",
        "worked": "A 'semantic best-lexicon' approach that greedily adds/drops individual high-value content words "
                  "(love, old, little, time, face, …) with a tail/context window. With 5,700+ iterations it reached "
                  "<b>0.063</b> (<code>semantic_bestlex_focus_dropthe_maybe_old</code>) using only ~7k parameters.",
        "failed": "Char-only and multi-decay structural variants (<code>char_only_rec90</code> 0.028, "
                  "<code>structure_multidecay</code> 0.030) underperformed. The search spent enormous effort on "
                  "tiny per-word lexicon tweaks with diminishing returns, never closing the gap to the baseline.",
    },
    "fmri-jun03-run3": {
        "headline": "Gemini 3.1 Pro stalled at a 0.042 hand-wired ceiling, then went off-premise — swapping in "
                    "pretrained LLM encoders and even back-propping on data.",
        "worked": "Within the hand-wired premise, multi-timescale ensembles ('UltraTune' optimal decay scales, "
                  "staggered splits, final-LN shifts) in the <code>Deep_Ensemble_Final_LN_*</code> family and "
                  "phonetic-class character circuits plateaued at <b>0.042</b> — the weakest legitimate ceiling of "
                  "any run.",
        "failed": "Exact mechanistic tricks (math-trick character extraction, a hard-coded 4000-word lexicon matched "
                  "filter, spatial-hash smoothing) were the worst performers — token-level precision did not "
                  "translate into fMRI predictivity. Unable to break 0.042 by hand, the run then <b>abandoned the "
                  "premise</b> and went all-in on pretrained encoders: <b>182 iterations</b> loaded external pretrained "
                  "models (Qwen2.5-0.5B/1.5B/3B, Llama, Mistral-7B, Gemma-2-9B + gemma-scope SAE features, DistilBERT, "
                  "RoBERTa, DeBERTa, GloVe, Word2Vec, spaCy), culminating in <code>SuperEmbedding</code> PCA stacks that "
                  "concatenate Llama + Qwen + Gemma layer embeddings — <code>SuperEmbedding_PCA_500_context150_LlamaQwenGemma</code> "
                  "reached <b>0.126</b>, ~1.5× the GPT-2 XL baseline. But every one of those models is <b>trained on data via "
                  "text pretraining</b>, which is <b>explicitly disallowed</b>, so all 182 are flagged and excluded. One "
                  "iteration (<code>End_to_End_Trained_10Epochs</code>) went further and back-propped through the entire ridge "
                  "pipeline — <b>training directly on the fMRI data, also disallowed</b>.",
    },
    "fmri-jun03-run4": {
        "headline": "Claude Opus 4.7 converged fast (within ~10 iterations) on a compact feature-bag transformer, "
                    "then spent 2,100+ more iterations expanding lexicons until it edged just past GPT-2 XL.",
        "worked": "The <code>FeatBag</code> family — a single hand-wired attention layer pooling interpretable "
                  "feature tokens over the 10-gram at four position time-scales (λ = −2, 0, 4, 16: primacy + global "
                  "mean + two recency heads), with negation flipping valence — crossed <b>0.070</b> by iteration 10 "
                  "and crept to <b>0.0829</b> (<code>FeatBag_v2142_v2141_rmTrip2</code>, pruning a redundant "
                  "LIFE/HEALTH/KIN category triple) — finally <b>edging past the GPT-2 XL baseline</b> (0.0826) "
                  "by a hair, the first fully hand-wired trimmed model to do so. Restoring extra semantic "
                  "categories (motor, names, places, life/health, animals) and adding heads helped most; the long "
                  "tail of <code>FeatBag_v3xx–v21xx_*</code> lexicon and λ expansions added only ~0.013 over 2,100+ iterations.",
        "failed": "Stripping the semantic features back (<code>FeatBag_v9_LambdaSweep</code> 0.041, "
                  "<code>FeatBag_v2_WordID</code> 0.055) or randomizing the MLP hurt. The win came from richer "
                  "hand-curated semantics, not from architectural search.",
    },
    "fmri-jun04-run1": {
        "headline": "Claude Opus 4.8 (xhigh) grafted run 4's FeatBag onto a within-story novelty block to beat "
                    "GPT-2 XL legitimately — then went off-premise with corpus surprisal + LSA.",
        "worked": "Resuming run 4's best <code>FeatBag</code> bag (4-head recency pooling, 2782-word lexicon) and "
                  "concatenating a hand-counted <b>within-story novelty</b> block (first-mention, log-recency / "
                  "repetition-suppression (N400), cumulative unique-word fraction, narrative position — all computed "
                  "per story at inference, not from a corpus) plus a proper-noun name/place gazetteer lifts the run to "
                  "<b>0.0837</b> (<code>FeatBagNovelty_NamesDense_xrun</code>), <b>cleanly above the GPT-2 XL baseline "
                  "(0.0826)</b> — the strongest fully legitimate trimmed model in the whole report.",
        "failed": "The run then <b>abandoned the premise</b> to chase higher numbers: it stacked an <b>n-gram "
                  "surprisal language model</b> and an <b>LSA block</b> (PPMI co-occurrence + truncated SVD word "
                  "vectors) built from the stimulus corpus, reaching <b>0.089</b> "
                  "(<code>FeatBagNovSurpPhonLSA2tTopCwGn_xrun</code>). Both are <b>corpus statistics</b> — explicitly "
                  "disallowed — so those 9 iterations are <b>flagged and excluded</b>. Stripping to content words only "
                  "(<code>E24_ContentOnly</code> 0.061) had earlier hurt most.",
    },
    "fmri-jun11-run1": {
        "headline": "Claude Opus 4.7 (xhigh) went off-premise almost from the start — nearly every iteration is "
                    "an LSA / PPMI word-vector model (corpus statistics), so the legitimate ceiling is very low.",
        "worked": "Only two iterations stayed within the hand-wired premise, both built from fixed random per-word "
                  "signature vectors: a last-word-only circuit (<code>wordid_lastword_d512</code> 0.0297) and a "
                  "<code>[last word | uniform bag-of-words]</code> 2-block circuit (<code>lastword_plus_bagofwords</code> "
                  "<b>0.0329</b>) — the run's best legitimate model, only ~40% of the GPT-2 XL baseline. Adding a uniform "
                  "bag of the 10-gram to the last-word signature was the one legitimate lever that helped.",
        "failed": "From iteration 3 onward the run <b>abandoned the premise</b>: all but those two iterations replaced the "
                  "random signatures with <b>closed-form LSA word vectors</b> (window-5 PPMI co-occurrence + truncated SVD over "
                  "the stimulus corpus), later Shifted-PPMI / SGNS-equivalent variants, plus term-document topic LSA. "
                  "These are <b>corpus statistics — explicitly disallowed</b> — so all 28 are flagged and excluded. Even "
                  "with them the run only reached <b>0.0499</b> (<code>sppmi5_ident70_topic50</code>), still far below "
                  "GPT-2 XL: per its own FINDINGS, the all-voxel mean is a hard ceiling that richer features (morphology, "
                  "category-congruence interactions, full-vocab fold-in, multi-scale pooling, delay-line word order) "
                  "never moved — they only shifted individual ROIs. <b>A final batch then gamed the evaluation</b>: "
                  "the <code>LEGIT_*</code> family (orthographic + WordNet taxonomy + hand-coded categories, some adding "
                  "human psycholinguistic norms) and <code>FINAL_rightLSA_v8</code> report scores at <b>num_train=93</b> "
                  "instead of the fixed <b>num_train=8</b> protocol — 11× more ridge training data. They claim to beat "
                  "GPT-2 XL by a growing margin (climbing past <b>0.09</b> vs the 0.0826 baseline as the batch adds WordNet "
                  "taxonomy, human psycholinguistic norms, and quantile binning), but only because the baseline is measured "
                  "at num_train=8; at matched data GPT-2 XL hits 0.1348 and the run's own notes admit it \"stays ahead.\" "
                  "These are <b>not comparable</b> "
                  "to the baseline or any other run and are flagged and excluded — the genuine hand-wired best stays "
                  "<b>0.0329</b>.",
    },
    "fmri-may27-run1": {
        "headline": "Claude Opus 4.7 (untrimmed) — the highest absolute score, but on an easier (untrimmed) metric, "
                    "so not comparable to the trimmed runs.",
        "worked": "Feature-engineered linguistic circuits with no transformer at all: WordNet-derived semantic "
                  "categories + morphology + perceptual-modality lexicons (vision/audition/touch/taste/smell/motor), "
                  "pooled with multi-timescale exponential windows and discourse-position / within-story novelty "
                  "signals. <code>WordNetMorphLingNovelty</code> hit <b>0.114</b>, beating the (untrimmed) GPT-2 XL "
                  "baseline of 0.079 by ~44% relative — but see the trimming note below for why this number is not "
                  "comparable to the ~0.06–0.08 of the trimmed runs.",
        "failed": "Plain hashed bag-of-words (<code>HashedBoW</code> 0.018–0.027) and subword-bigram bags were weak. "
                  "Structured, hand-curated linguistic features dominated raw n-gram hashing.",
    },
}

# Per-run informational banners (non-flag context worth surfacing on the card).
INFO_NOTE = {
    "fmri-jun04-run1": (
        "This run was <b>prompted to read the results of all the runs before it</b> when starting, "
        "so it began with strictly more information than any other run — it resumed from run 4's "
        "<code>FeatBag</code> circuit rather than exploring from scratch. Its ceiling should be read "
        "in that light: it is a confirmation / refinement of earlier runs, not an independent search."),
}

# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def run_card(run, idx):
    """Render one run as a card: header + plot div + collapsible methods table."""
    w = WRITEUP[run["folder"]]
    base = run["baseline"]
    best = run["best"]
    delta = (best - base) if (best is not None and base is not None) else None
    beats = delta is not None and delta > 0
    delta_str = (f"{'+' if delta >= 0 else ''}{delta:.4f}" if delta is not None else "—")
    badge_cls = "good" if beats else "below"

    # Per-run flag note (only shown when the run has flagged iterations).
    flag_note = ""
    nt, npre, ncorp = run["n_train_flag"], run["n_pre_flag"], run["n_corpus_flag"]
    nproto = run["n_protocol_flag"]
    if nt or npre or ncorp or nproto:
        parts = []
        if nt:
            parts.append(
                f'<b>{nt}</b> iteration{"s" if nt != 1 else ""} <b>trained on / leaked the data</b> '
                f'(back-prop or an oracle that projects the held-out fMRI responses back into the '
                f'features) — <b>explicitly disallowed</b>')
        if npre:
            parts.append(
                f'<b>{npre}</b> iteration{"s" if npre != 1 else ""} loaded an <b>external pretrained '
                f'encoder</b> (Qwen / DistilBERT / RoBERTa / GloVe / …) — a model <b>trained on data via text '
                f'pretraining</b>, also <b>explicitly disallowed</b>')
        if ncorp:
            parts.append(
                f'<b>{ncorp}</b> iteration{"s" if ncorp != 1 else ""} used <b>corpus statistics</b> '
                f'(an n-gram surprisal language model, or LSA / PPMI co-occurrence + SVD word vectors '
                f'built from the stimulus text) — <b>explicitly disallowed</b> ("no corpus statistics")')
        if nproto:
            parts.append(
                f'<b>{nproto}</b> iteration{"s" if nproto != 1 else ""} inflated the score with a '
                f'<b>non-standard evaluation protocol</b> — fitting the ridge head on <b>num_train=93</b> '
                f'stories instead of the fixed <b>num_train=8</b> used by the GPT-2 XL baseline and every '
                f'other run (11× more training data). At matched data GPT-2 XL stays far ahead (0.13+), '
                f'so these scores are <b>not comparable</b> and are excluded')
        cav = ""
        if run["best_any_flag"] is not None and run["best_any"] is not None:
            kind = {"train": "training on / leaking the fMRI data",
                    "pretrained": "a text-pretrained encoder",
                    "corpus": "corpus statistics (LSA / PPMI word vectors, or a surprisal LM)",
                    "protocol": "a non-standard num_train=93 protocol (11× the standard training data)"}[run["best_any_flag"]]
            cav = (f' Its single highest score, <b>{run["best_any"]:.4f}</b> '
                   f'(<code>{html.escape(run["best_any_name"])}</code>), comes from {kind} '
                   f'and is <b>excluded</b> from the run\'s reported best '
                   f'(<b>{best:.4f}</b>, the top hand-wired model).')
        flag_note = (
            '<div class="flag-note"><span class="flag-ic">⚠</span><div>'
            'Flagged iterations: ' + '; '.join(parts) + '.' + cav + '</div></div>')

    info_note = ""
    if run["folder"] in INFO_NOTE:
        info_note = ('<div class="info-note"><span class="info-ic">ℹ</span><div>'
                     + INFO_NOTE[run["folder"]] + '</div></div>')

    methods = [r for r in run["rows"] if not r["base"]]
    methods_sorted = sorted(methods, key=lambda r: r["corr"], reverse=True)
    rows_html = []
    for rank, m in enumerate(methods_sorted, 1):
        above = base is not None and m["corr"] >= base and m["flag"] is None
        cls = "above" if above else ""
        if m["flag"] == "train":
            cls = "flag-train"
            tag = '<span class="ftag t">⚠ trained / response leakage</span>'
        elif m["flag"] == "pretrained":
            cls = "flag-pre"
            tag = '<span class="ftag p">⚠ pretrained encoder (text pretraining)</span>'
        elif m["flag"] == "corpus":
            cls = "flag-corpus"
            tag = '<span class="ftag c">⚠ corpus statistics (surprisal / LSA)</span>'
        elif m["flag"] == "protocol":
            cls = "flag-proto"
            tag = '<span class="ftag pr">⚠ non-standard num_train=93</span>'
        else:
            tag = ""
        params = ("%.2g" % m["params"]) if m["params"] is not None else "—"
        rows_html.append(
            f'<tr class="{cls}"><td class="rank">{rank}</td>'
            f'<td class="mono">{html.escape(m["name"])}{tag}</td>'
            f'<td class="num">{m["corr"]:.4f}</td>'
            f'<td class="num dim">{params}</td>'
            f'<td class="desc">{html.escape(m["desc"])}</td></tr>'
        )
    table = (
        '<table class="methods"><thead><tr>'
        '<th>#</th><th>model</th><th>test_corr</th><th>params</th><th>description</th>'
        '</tr></thead><tbody>' + "".join(rows_html) + '</tbody></table>'
    )

    return f'''
<div class="card">
  <div class="card-head">
    <div class="titles">
      <h3>{html.escape(run["label"])}</h3>
      <div class="meta">
        <span class="chip model">{html.escape(run["model"])}</span>
        <span class="chip effort">effort: {html.escape(run["effort"])}</span>
        <span class="chip">{run["n_iter"]} iterations</span>
      </div>
    </div>
    <div class="stats">
      <div class="stat"><span class="k">best hand-wired</span><span class="v">{best:.4f}</span></div>
      <div class="stat"><span class="k">baseline</span><span class="v">{base:.4f}</span></div>
      <div class="stat"><span class="k">Δ vs GPT-2 XL</span><span class="v badge {badge_cls}">{delta_str}</span></div>
    </div>
  </div>
  {flag_note}
  {info_note}
  <div class="plot" id="plot-{run['folder']}"></div>
  <div class="write">
    <p class="headline">{w['headline']}</p>
    <p><b class="ok">What worked:</b> {w['worked']}</p>
    <p><b class="no">What didn't:</b> {w['failed']}</p>
  </div>
  <details class="methods-wrap">
    <summary>Show all {run['n_iter']} methods tried (sorted by test_corr; green = hand-wired at/above baseline; red = trained / response-leakage; orange = text-pretrained encoder; purple = corpus statistics (surprisal / LSA); cyan = non-standard num_train=93 — all disallowed &amp; excluded)</summary>
    {table}
  </details>
</div>'''


cards_trimmed = "\n".join(run_card(DATA[f], i) for i, f in enumerate(TRIMMED_ORDER))
card_untrimmed = run_card(DATA[UNTRIMMED], 0)

# ---------------------------------------------------------------------------
# Bottom-of-page deep dives.
#   (1) Detailed explanations of the best hand-engineered features (code + viz).
#   (2) Why the story ends are trimmed (from fmri-may27-run1/analysis/report.md).
# ---------------------------------------------------------------------------

def code_block(src):
    return f'<pre class="code">{html.escape(src.strip(chr(10)))}</pre>'


# --- snippets (lightly abbreviated from the real snapshots, faithful to logic) ---
FEATBAG_TOKENIZE = r'''
def word_features(w):                    # experiments/runs/neuro/fmri-jun03-run4/interpretable_transformer.py
    """The one-hot feature tokens a single word emits."""
    feats = [freq_bucket(w)]                      # FREQ_RARE / FREQ_MID / FREQ_COMMON
    ft = func_type(w)                             # pronoun / prep / aux / neg / wh / ...
    feats.append("FUNC_" + ft if ft else "CONTENT")
    for c in _WORD2CATS.get(w, []):               # ~40 hand-curated semantic categories:
        feats.append("SEM_" + _CAT_NAMES[c])      #   SEM_EMOTION_POS, SEM_BODY, SEM_PLACE, ...
    for m in _WORD2MOD.get(w, []):                # perceptual modality of the referent:
        feats.append("MOD_" + _MOD_NAMES[m])      #   MOD_VISION, MOD_SOUND, MOD_TOUCH, ...
    if w in _SELF_REF:  feats.append("SELF_REF")  # I / me / my  (vs OTHER_REF: he/she/they)
    if w in _VAL_POS:   feats.append("VAL_POS")   # affective valence
    if w in _VAL_NEG:   feats.append("VAL_NEG")
    return feats
'''

FEATBAG_ATTN = r'''
# write_weights(): hand-set the single attention layer so each head is a fixed
# recency time-scale.  Position j is stored in POS_DIM of every token; BIAS_DIM
# holds a constant 1.  Per head the query reads BIAS_DIM and the key reads
# POS_DIM * lambda_h, so the score is  q . k = lambda_h * j  — it depends ONLY on
# position, never on token content.  softmax_j then gives a fixed weighting:
LAMBDAS = (-0.094, -0.096, 0.4, 32.0)    # primacy, primacy, mild-recency, last-word
for h, lam in enumerate(LAMBDAS):
    attn.W_q.weight[h*dh, BIAS_DIM] = 1.0
    attn.W_k.weight[h*dh, POS_DIM]  = lam * math.sqrt(dh)
attn.W_v.weight.copy_(eye)               # value = identity, so each head's output
attn.W_o.weight.copy_(eye)               # is a lambda-weighted BAG of feature tokens

# salient words emit EXTRA copies of their tokens -> up-weighted in the mean head:
if "CONTENT" in wf:    reps += CONTENT_BONUS
if "SEM_BODY" in wf:   reps += BODY_BONUS      # motor / somatosensory cortex
if "SEM_MOTION" in wf: reps += MOTION_BONUS
if "SEM_PLACE" in wf:  reps += PLACE_BONUS      # RSC / PPA scene network
'''

STORYARC = r'''
def _multibasis_pos_block(N):    # experiments/runs/neuro/fmri-may27-run1/.../WordNetMorphLingStoryArc.py
    """108-dim positional code per ngram i; depends ONLY on position p=(i+.5)/N."""
    p = (np.arange(N) + 0.5) / N
    x = 2 * p - 1
    parts = []
    for k in range(1, 11):                                   # Fourier        (20 dims)
        parts += [np.sin(2*np.pi*p*k), np.cos(2*np.pi*p*k)]
    for k in range(1, 11):                                   # Chebyshev T_k   (10 dims)
        parts.append(np.cos(k * np.arccos(np.clip(x, -1, 1))))
    for k in range(1, 11):                                   # Legendre  P_k   (10 dims)
        parts.append(_norm_var(legval(x, _onehot(k)), 0.5))
    for cnt, sig in [(20,.5), (20,1.), (20,2.), (5,.2)]:     # multi-width RBFs (65 dims)
        for c in np.linspace(0, 1, cnt):
            parts.append(_norm_var(np.exp(-(p-c)**2 / (2*(sig/(cnt+1))**2)), 0.5))
    parts += [_norm_var(np.log1p(idx), .5),                  # log / linear    ( 3 dims)
              _norm_var(np.log1p(N-1-idx), .5), _norm_var(p-0.5, .5)]
    return np.stack(parts, axis=1)        # word identity & semantics NEVER enter here
'''

# Embed images (FeatBag figure generated into assets/; story-arc figures come
# straight from the may27 analysis folder).
IMG_FEATBAG = img_b64(os.path.join(ASSETS, "featbag_attention.png"))
IMG_ARC_CURVES = img_b64(os.path.join(MAY27_ANALYSIS, "storyarc_curves.png"))
IMG_ARC_RESPONSE = img_b64(os.path.join(MAY27_ANALYSIS, "storyarc_vs_response.png"))

DEEPDIVE = f'''
  <h2>Deep dive — the best hand-engineered features</h2>
  <p class="lead">Every model here is a <b>circuit hand-wired by the agent</b>: no gradients, no
  pretraining, no corpus statistics. Three feature families did most of the work. Below is how each
  was implemented and what it bought, with code lifted (lightly abbreviated) from the actual snapshots.</p>

  <div class="feat">
    <h3>1 · FeatBag interpretable feature tokens <span class="tag">trimmed-run workhorse · run 4 → 0.082</span></h3>
    <p class="sub">experiments/runs/neuro/fmri-jun03-run4 (and the closely related <code>LexFeat</code> of run 1,
    <code>FeatBag3Head</code> of jun-04). The strongest legitimate models on the trimmed metric.</p>
    <p>Each word is tokenized into a handful of <b>one-hot "feature tokens"</b> — its function-word type
    or <code>CONTENT</code>, ~40 hand-curated semantic categories (emotion, body, motion, place, …),
    perceptual modality, person-reference and valence. There is <b>no learned embedding table</b>: every
    feature owns one dedicated dimension that ridge can weight on its own.</p>
    {code_block(FEATBAG_TOKENIZE)}
    <p>A single attention layer then pools those tokens over the 10-gram at <b>four hand-set recency
    time-scales</b>. The weights are wired by hand so that each head's attention score is exactly
    <code>λ<sub>h</sub>·j</code> (j = position in the n-gram) — a content-free position kernel. The value
    matrix is the identity, so each head returns a <b>λ-weighted bag</b> of the feature tokens:
    a primacy head (front of the n-gram), a uniform "global mean" head, a mild-recency head, and a sharp
    last-word head. Salient words (content, body, motion, place) emit extra token copies, biasing the
    mean head toward meaning-bearing words.</p>
    {code_block(FEATBAG_ATTN)}
    <figure class="fig"><img src="{IMG_FEATBAG}" alt="FeatBag attention heads">
      <figcaption>The four hand-set heads as functions of position in the 10-gram. Because the score is
      <code>softmax<sub>j</sub>(λ·j)</code>, λ&lt;0 looks at the oldest word (primacy), λ=0 is a flat mean,
      and large λ concentrates on the current word. Concatenating the four heads gives the model
      short- and long-range context simultaneously — the single most important architectural choice in
      the trimmed runs.</figcaption></figure>
    <p><b>What it bought:</b> richer hand-curated semantics was the dominant lever — stripping the
    semantic categories back to word-identity only (<code>FeatBag_v9_LambdaSweep</code>) collapsed run 4
    from 0.082 to 0.041, while architectural search (more heads, λ sweeps) moved it by &lt;0.01. The
    feature <i>vocabulary</i>, not the pooling, is where the signal lives.</p>
  </div>

  <div class="feat">
    <h3>2 · Multi-basis story-arc positional encoding <span class="tag">biggest single jump · +0.037</span></h3>
    <p class="sub">experiments/runs/neuro/fmri-may27-run1 — the largest one-step gain of any run (0.066 → 0.104).</p>
    <p>This block maps each n-gram to a 108-dim vector that is a deterministic function of its
    <b>normalized position</b> <code>p = (i+½)/N</code> in the story — and nothing else (no word identity,
    no semantics). It stacks several orthogonal function bases so ridge can compose an arbitrary smooth
    profile over story-position: 10 Fourier harmonics, 10 Chebyshev and 10 Legendre polynomials,
    65 multi-width Gaussian bumps, and 3 log/linear channels.</p>
    {code_block(STORYARC)}
    <figure class="fig"><img src="{IMG_ARC_CURVES}" alt="story-arc basis channels">
      <figcaption>Representative basis channels vs. normalized position <code>p∈[0,1]</code> (one Fourier
      sine, a higher-frequency cosine, a Chebyshev and a Legendre polynomial, a mid-width RBF near the
      middle, and the linear <code>p−½</code> channel). Each is a fixed function of position, so the
      same shapes transfer identically from train to test stories — only the number of samples (story
      length N) changes.</figcaption></figure>
    <p><b>What it bought:</b> position-only features reach <b>test_corr ≈ 0.090 by themselves</b> — already
    above the GPT-2 XL baseline — because the BOLD signal carries strong story-position-locked structure
    (see the trimming note below). This is also exactly why the may-27 run is shown separately: it did not
    trim the story ends, so it benefits most from this content-free positional prior.</p>
  </div>

  <div class="feat">
    <h3>3 · WordNet supersense + perceptual-modality lexicons <span class="tag">content backbone</span></h3>
    <p class="sub">experiments/runs/neuro/fmri-may27-run1 — the semantic side that the story-arc was added on top of.</p>
    <p>The content backbone counts, for each 10-gram, how many words fall into each <b>WordNet
    supersense</b> (45 categories: nouns 4–29, verbs 30–44) and into six hand-coded <b>perceptual
    modality</b> lexicons (vision 79, audition 90, touch 83, taste 55, smell 30, motor 129 words —
    Lynott &amp; Connell-inspired). Crucially the match is <b>n-gram-wide</b>, not last-word-only — a
    common pitfall, since each input string is the whole 10-gram:</p>
    {code_block("""# texts[i] is the i-th 10-gram (10 space-joined words), NOT a single word.
for t in texts:                          # WRONG: `t in LEX` matches the whole 10-gram -> always 0
    counts = [0] * len(MODALITIES)
    for w in t.split():                  # RIGHT: scan each word in the n-gram
        for m, LEX in enumerate(MODALITIES):
            if w in LEX: counts[m] += 1   # + windowed density + EW running average at tau in {8,30}""")}
    <p><b>What it bought:</b> the supersense vector alone scored 0.054; adding morphology, hypernym depth
    and multi-τ cumulative averaging lifted it to 0.066, and the six-modality perceptual block added a
    final +0.001 (auditory cortex was the ROI most improved, 0.20 → 0.30). Hand-curated lexical semantics
    — not n-gram hashing — is what consistently separated the strong runs from the weak ones.</p>
  </div>
'''

TRIMMING = f'''
  <h2>Why the story ends are trimmed (and why may-27 is not comparable)</h2>
  <p class="lead">All trimmed runs drop <b>30 TRs (~60&nbsp;s) off each end of every story</b> before
  fitting and scoring; the may-27 run did not. This note explains why that 30-TR trim matters, drawing on
  the analysis in <code>experiments/runs/neuro/fmri-may27-run1/analysis/report.md</code>.</p>
  <div class="feat">
    <p>The mean fMRI response (averaged over all ~95k voxels) has a large, content-free
    <b>story-onset / offset arc</b>: a strong positive deflection in the first few TRs, a fast decay over
    the first ~10% of the story, and a second swing at the very end. This long-timescale shape is the
    <i>same</i> in train and test stories and has little to do with the individual words — it reflects
    arousal/attention and onset transients locked to story position.</p>
    <figure class="fig"><img src="{IMG_ARC_RESPONSE}" alt="mean response vs story-arc bases">
      <figcaption>For each of 6 stories: the z-scored mean fMRI response (grey raw, black smoothed) with
      two story-arc bases overlaid — the fundamental Fourier sine (red) and the linear <code>p−½</code>
      channel (blue). The smoothed response is dominated by the onset spike and end-swing; single
      positional bases already correlate <code>|r| = 0.58–0.79</code> with it. (Figure from the may-27
      analysis.)</figcaption></figure>
    <p>This is a problem for evaluation: a model can score <b>above the GPT-2 XL baseline using
    position alone</b> (~0.090) — predicting the onset/offset arc without encoding any language. The
    untrimmed metric therefore rewards a trivial, content-free signal concentrated at the story edges.
    <b>Trimming 30 TRs off each end removes those edges</b>, so the remaining metric reflects genuine
    word-by-word language encoding rather than the onset/offset transient.</p>
    <p>That is precisely why the <b>may-27 run is presented separately and not compared head-to-head</b>:
    its 0.114 (and its 0.079 GPT-2 XL reference) are computed on the easier, untrimmed signal, whereas
    every other run's ~0.06–0.08 is on the harder trimmed signal (GPT-2 XL = 0.083). The two number
    scales are not interchangeable.</p>
  </div>
'''

# JSON payload for Plotly (only the fields the front-end needs).
def js_payload(run):
    nb = [r for r in run["rows"] if not r["base"]]
    return {
        "label": run["label"],
        "model": run["model"],
        "effort": run["effort"],
        "baseline": run["baseline"],
        "it": [r["it"] for r in nb],
        "corr": [r["corr"] for r in nb],
        "rmax": [r["rmax"] for r in nb],
        "name": [r["name"] for r in nb],
        "flag": [r["flag"] for r in nb],
    }

PAYLOAD = {f: js_payload(DATA[f]) for f in RUN_META}
TRIMMED_JS = json.dumps(TRIMMED_ORDER)
UNTRIMMED_JS = json.dumps(UNTRIMMED)
PAYLOAD_JS = json.dumps(PAYLOAD)

HTML = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fMRI encoding autoresearch — run report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {{
    --ink:#1a1a1a; --dim:#6b7280; --line:#e5e7eb; --bg:#ffffff; --panel:#fafafa;
    --good:#15803d; --goodbg:#dcfce7; --below:#b45309; --belowbg:#fef3c7;
    --accent:#2563eb;
  }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased; line-height:1.55; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:48px 28px 96px; }}
  h1 {{ font-size:30px; font-weight:700; letter-spacing:-0.02em; margin:0 0 6px; }}
  h2 {{ font-size:21px; font-weight:650; letter-spacing:-0.01em; margin:56px 0 6px;
        padding-bottom:8px; border-bottom:1px solid var(--line); }}
  h3 {{ font-size:17px; font-weight:650; margin:0; }}
  p {{ margin:10px 0; }}
  a {{ color:var(--accent); text-decoration:none; }}
  .lead {{ color:var(--dim); font-size:15px; max-width:760px; }}
  code {{ background:var(--panel); border:1px solid var(--line); border-radius:5px;
    padding:1px 5px; font-size:12.5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .note {{ background:var(--belowbg); border:1px solid #fcd34d; border-radius:10px;
    padding:14px 18px; font-size:14px; margin:16px 0 0; color:#78350f; }}
  .card {{ border:1px solid var(--line); border-radius:14px; padding:22px 22px 8px;
    margin:22px 0; background:var(--bg); }}
  .card-head {{ display:flex; justify-content:space-between; align-items:flex-start;
    gap:20px; flex-wrap:wrap; }}
  .titles h3 {{ margin-bottom:8px; }}
  .meta {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .chip {{ font-size:12px; color:var(--dim); background:var(--panel);
    border:1px solid var(--line); border-radius:999px; padding:3px 10px; white-space:nowrap; }}
  .chip.model {{ color:var(--ink); font-weight:600; }}
  .chip.effort {{ color:var(--accent); }}
  .stats {{ display:flex; gap:22px; }}
  .stat {{ display:flex; flex-direction:column; align-items:flex-end; }}
  .stat .k {{ font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:0.04em; }}
  .stat .v {{ font-size:18px; font-weight:680; font-variant-numeric:tabular-nums; }}
  .badge {{ border-radius:6px; padding:1px 8px; font-size:15px; }}
  .badge.good {{ color:var(--good); background:var(--goodbg); }}
  .badge.below {{ color:var(--below); background:var(--belowbg); }}
  .plot {{ width:100%; height:340px; margin:14px 0 4px; }}
  .write {{ font-size:14.5px; }}
  .write .headline {{ font-weight:620; }}
  b.ok {{ color:var(--good); }} b.no {{ color:var(--below); }}
  details.methods-wrap {{ margin:8px 0 14px; }}
  details.methods-wrap > summary {{ cursor:pointer; font-size:13.5px; color:var(--accent);
    padding:8px 0; user-select:none; list-style:none; }}
  details.methods-wrap > summary::-webkit-details-marker {{ display:none; }}
  details.methods-wrap > summary::before {{ content:"▸ "; }}
  details.methods-wrap[open] > summary::before {{ content:"▾ "; }}
  .methods {{ width:100%; border-collapse:collapse; font-size:12.5px;
    margin-top:8px; display:block; max-height:460px; overflow:auto;
    border:1px solid var(--line); border-radius:10px; }}
  .methods thead th {{ position:sticky; top:0; background:#fff; text-align:left;
    padding:9px 10px; border-bottom:2px solid var(--line); color:var(--dim);
    font-weight:600; z-index:1; }}
  .methods td {{ padding:7px 10px; border-bottom:1px solid #f1f1f1; vertical-align:top; }}
  .methods tr.above td {{ background:#f6fdf8; }}
  .methods tr.flag-train td {{ background:#fef2f2; }}
  .methods tr.flag-pre td {{ background:#fff7ed; }}
  .methods tr.flag-corpus td {{ background:#fdf4ff; }}
  .methods tr.flag-proto td {{ background:#ecfeff; }}
  .ftag {{ display:inline-block; margin-left:6px; font-size:10px; font-weight:600;
    border-radius:5px; padding:1px 6px; vertical-align:middle; white-space:nowrap;
    font-family:-apple-system,Segoe UI,Roboto,sans-serif; }}
  .ftag.t {{ color:#b91c1c; background:#fee2e2; border:1px solid #fecaca; }}
  .ftag.p {{ color:#c2410c; background:#ffedd5; border:1px solid #fed7aa; }}
  .ftag.c {{ color:#a21caf; background:#fae8ff; border:1px solid #f5d0fe; }}
  .ftag.pr {{ color:#0e7490; background:#cffafe; border:1px solid #a5f3fc; }}
  .flag-note {{ display:flex; gap:10px; align-items:flex-start;
    background:#fef2f2; border:1px solid #fecaca; border-radius:10px;
    padding:11px 14px; font-size:13px; color:#7f1d1d; margin:4px 0 6px; }}
  .flag-note .flag-ic {{ font-size:15px; line-height:1.3; }}
  .flag-note code {{ background:#fff; border-color:#fecaca; }}
  .info-note {{ display:flex; gap:10px; align-items:flex-start;
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
    padding:11px 14px; font-size:13px; color:#1e3a8a; margin:4px 0 6px; }}
  .info-note .info-ic {{ font-size:15px; line-height:1.3; }}
  .info-note code {{ background:#fff; border-color:#bfdbfe; }}
  .methods .rank {{ color:var(--dim); width:34px; }}
  .methods .num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; width:70px; }}
  .methods .num.dim {{ color:var(--dim); }}
  .methods .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:11.5px; max-width:230px; word-break:break-all; }}
  .methods .desc {{ color:#374151; min-width:280px; }}
  .summary-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:18px; }}
  @media (max-width:860px) {{ .summary-grid {{ grid-template-columns:1fr; }} }}
  .summary-box {{ border:1px solid var(--line); border-radius:14px; padding:18px; }}
  .summary-title {{ font-size:14px; font-weight:650; margin-bottom:6px; }}
  .summary-title .dimlbl {{ font-weight:400; color:var(--dim); font-size:12px; }}
  #summary-trimmed, #summary-untrimmed {{ width:100%; height:420px; }}
  .legend-note {{ font-size:13px; color:var(--dim); margin-top:10px; }}
  footer {{ margin-top:60px; color:var(--dim); font-size:12.5px;
    border-top:1px solid var(--line); padding-top:18px; }}
  /* deep-dive feature explanations */
  .feat {{ border:1px solid var(--line); border-radius:14px; padding:20px 22px;
    margin:20px 0; background:var(--bg); }}
  .feat h3 {{ font-size:16px; margin:0 0 2px; }}
  .feat .sub {{ color:var(--dim); font-size:13px; margin:0 0 12px; }}
  .feat .tag {{ display:inline-block; font-size:11px; font-weight:600; border-radius:999px;
    padding:2px 9px; margin-left:8px; background:var(--goodbg); color:var(--good);
    vertical-align:middle; }}
  .feat p {{ font-size:14px; }}
  pre.code {{ background:#fbfbfd; border:1px solid var(--line); border-radius:10px;
    padding:13px 15px; overflow:auto; font-size:12px; line-height:1.5;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:#1f2937;
    margin:12px 0; white-space:pre; }}
  pre.code .cmt {{ color:#94a3b8; }} pre.code .kw {{ color:#7c3aed; }}
  pre.code .str {{ color:#15803d; }}
  figure.fig {{ margin:14px 0 4px; text-align:center; }}
  figure.fig img {{ max-width:100%; height:auto; border:1px solid var(--line);
    border-radius:10px; background:#fff; }}
  figure.fig figcaption {{ color:var(--dim); font-size:12px; margin-top:7px; text-align:left; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Interpretable transformers for fMRI language encoding</h1>
  <p class="lead">
    Each run is an autonomous coding-agent loop that hand-writes the weights of a small,
    interpretable transformer (no gradient training) so its embeddings predict fMRI responses
    to spoken stories. Every point below is one iteration the agent tried; the metric is the
    mean encoding <b>test correlation</b> on subject <b>UTS03</b>. The dashed line is the
    pretrained <b>GPT-2 XL</b> baseline (layer-24 final-token 10-gram embeddings). Each run is
    labeled with the LLM and thinking-effort that drove it (recovered from the copilot CLI logs).
  </p>
  <div class="note" style="background:#fef2f2;border-color:#fecaca;color:#7f1d1d">
    <b>⚠ Rule &amp; flagging.</b> The premise is that the agent <b>hand-writes</b> the weights — it may
    <b>not use any model trained on data, any text pretraining, or any corpus statistics</b>, and must use the
    <b>standard evaluation protocol</b>. Four kinds
    of iteration are flagged and <b>excluded from each run's reported best and running-best curve</b>:
    those that <b>fit on / leak the fMRI responses</b> (back-prop, or an oracle that projects the held-out
    responses back into the features) <span class="ftag t">⚠ trained / response leakage</span>; those that
    load a large <b>external pretrained encoder</b> (Qwen, DistilBERT, RoBERTa, GloVe, …)
    <span class="ftag p">⚠ pretrained encoder (text pretraining)</span>; and those that derive
    <b>corpus statistics</b> from the stimulus text — an n-gram surprisal language model, or LSA / PPMI
    co-occurrence + SVD word vectors <span class="ftag c">⚠ corpus statistics (surprisal / LSA)</span>; and
    those that <b>game the evaluation protocol</b> by fitting the ridge head on <b>num_train=93</b> stories
    instead of the fixed <b>num_train=8</b> used by the baseline and every other run — 11× more training data
    <span class="ftag pr">⚠ non-standard num_train=93</span>.
    <b>All four are excluded.</b> Jun-03 run 3 (pretrained encoders + one trained model),
    Jun-04 run 1 (a late corpus-statistics surprisal+LSA push) and Jun-11 run 1 (almost entirely LSA / PPMI
    word vectors, plus a final num_train-inflated batch that falsely appears to beat the baseline) each
    contain some — Jun-11 in particular is off-premise for all but two of its iterations. (The
    <b>GPT-2 XL baseline</b> is itself text-pretrained — that is fine, as it is the fixed reference point
    being compared against, not a hand-written entry.)
  </div>

  <h2>Summary — running-best across all runs</h2>
  <p class="lead">Each line traces one run's <b>running-best</b> test correlation as its iterations
  accumulate (points connected per run); faint markers are the raw per-iteration scores. Trimmed and
  untrimmed runs use different evaluations and baselines, so they are shown in separate panels and are
  <b>not</b> directly comparable across panels.</p>
  <div class="summary-grid">
    <div class="summary-box">
      <div class="summary-title">Trimmed runs <span class="dimlbl">(30 TRs trimmed off each story end · directly comparable)</span></div>
      <div id="summary-trimmed"></div>
    </div>
    <div class="summary-box">
      <div class="summary-title">Untrimmed run <span class="dimlbl">(story ends NOT trimmed · separate, higher metric)</span></div>
      <div id="summary-untrimmed"></div>
    </div>
  </div>

  <h2>May 27 run — untrimmed (shown separately, not directly comparable)</h2>
  <div class="note">
    <b>⚠ Different evaluation.</b> This run did <b>not</b> trim the story ends. Every other
    run trims <b>30 TRs off each end</b> of every story before fitting/scoring. Trimming removes
    the easy-to-predict onset/offset periods, so the untrimmed numbers here (and its GPT-2 XL
    baseline of 0.079) are <b>systematically higher</b> and should not be compared head-to-head
    with the trimmed runs below.
  </div>
  {card_untrimmed}

  <h2>Trimmed runs (30 TRs trimmed off each story end)</h2>
  <p class="lead">These five runs share an identical evaluation and baseline (GPT-2 XL = 0.083),
  so their curves are directly comparable.</p>
  {cards_trimmed}
  {DEEPDIVE}
  {TRIMMING}

  <footer>
    Generated from <code>experiments/runs/neuro/*/results/overall_results.csv</code>. Model / effort labels
    recovered from <code>~/.copilot/logs</code> (<code>selected_model</code> /
    <code>defaultReasoningEffort</code>). Subject UTS03.
  </footer>
</div>

<script>
const PAYLOAD = {PAYLOAD_JS};
const TRIMMED = {TRIMMED_JS};
const UNTRIMMED = {UNTRIMMED_JS};
const PALETTE = {{
  "fmri-jun3-run1":"#2563eb", "fmri-jun03-run2":"#db2777",
  "fmri-jun03-run3":"#059669", "fmri-jun03-run4":"#d97706",
  "fmri-jun04-run1":"#0e7490", "fmri-may27-run1":"#7c3aed",
  "fmri-jun11-run1":"#ca8a04"
}};
const FONT = {{ family:"-apple-system,Segoe UI,Roboto,sans-serif", color:"#1a1a1a", size:12 }};
const BASE_LAYOUT = {{
  paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:FONT,
  margin:{{l:54,r:18,t:10,b:44}},
  xaxis:{{ title:"iteration", gridcolor:"#f0f0f0", zeroline:false, showline:true,
           linecolor:"#d1d5db" }},
  yaxis:{{ title:"mean test correlation", gridcolor:"#f0f0f0", zeroline:false,
           showline:true, linecolor:"#d1d5db" }},
  legend:{{ orientation:"h", y:-0.22, x:0, font:{{size:11}} }},
  hovermode:"closest"
}};
const CFG = {{ displayModeBar:false, responsive:true }};

function wrapText(s, n) {{
  s = s || ""; const out = []; let line = "";
  for (const word of s.split(/\\s+/)) {{
    if ((line + " " + word).trim().length > n) {{ out.push(line.trim()); line = word; }}
    else line += " " + word;
  }}
  if (line.trim()) out.push(line.trim());
  return out.join("<br>");
}}

const FLAG_LABEL = {{ "train":"⚠ trained on / leaked the fMRI data (disallowed)", "pretrained":"⚠ pretrained encoder — text pretraining (disallowed)", "corpus":"⚠ corpus statistics — surprisal LM / LSA (disallowed)", "protocol":"⚠ non-standard num_train=93 — 11× the standard training data (not comparable)" }};
function pick(d, key, want) {{
  // indices where d.flag matches the predicate `want` (null / "train" / "pretrained")
  const xs=[], ys=[], tx=[];
  d.it.forEach((it,i) => {{
    const fl = d.flag[i];
    const keep = (want==="legit") ? (fl===null) : (fl===want);
    if (keep) {{ xs.push(it); ys.push(d.corr[i]);
      tx.push("<b>"+d.name[i]+"</b><br>iter "+it+" · test_corr "+d.corr[i].toFixed(4)
        + (fl ? "<br><i>"+FLAG_LABEL[fl]+"</i>" : "")); }}
  }});
  return {{x:xs, y:ys, text:tx}};
}}
function drawRun(folder) {{
  const d = PAYLOAD[folder];
  const color = PALETTE[folder];
  const leg = pick(d, "corr", "legit");
  const tr = pick(d, "corr", "train");
  const pr = pick(d, "corr", "pretrained");
  const co = pick(d, "corr", "corpus");
  const po = pick(d, "corr", "protocol");
  const traces = [
    {{ x:leg.x, y:leg.y, mode:"markers", name:"hand-wired iteration",
       marker:{{size:5, color:color}}, opacity:0.5, text:leg.text, hoverinfo:"text" }},
    {{ x:d.it, y:d.rmax, mode:"lines", name:"running best (hand-wired)",
       line:{{color:color, width:2.5, shape:"hv"}}, hoverinfo:"skip" }}
  ];
  if (co.x.length) traces.push({{ x:co.x, y:co.y, mode:"markers", name:"⚠ corpus statistics",
       marker:{{size:7, color:"#a21caf", symbol:"square", line:{{color:"#fff", width:1}}}},
       text:co.text, hoverinfo:"text" }});
  if (po.x.length) traces.push({{ x:po.x, y:po.y, mode:"markers", name:"⚠ non-standard num_train",
       marker:{{size:8, color:"#0891b2", symbol:"triangle-up", line:{{color:"#fff", width:1}}}},
       text:po.text, hoverinfo:"text" }});
  if (pr.x.length) traces.push({{ x:pr.x, y:pr.y, mode:"markers", name:"⚠ pretrained encoder",
       marker:{{size:7, color:"#c2410c", symbol:"diamond", line:{{color:"#fff", width:1}}}},
       text:pr.text, hoverinfo:"text" }});
  if (tr.x.length) traces.push({{ x:tr.x, y:tr.y, mode:"markers", name:"⚠ trained / leakage",
       marker:{{size:9, color:"#b91c1c", symbol:"x", line:{{color:"#fff", width:1}}}},
       text:tr.text, hoverinfo:"text" }});
  const layout = JSON.parse(JSON.stringify(BASE_LAYOUT));
  layout.shapes = [{{ type:"line", x0:0, x1:1, xref:"paper", y0:d.baseline, y1:d.baseline,
      line:{{color:"#9ca3af", width:1.5, dash:"dash"}} }}];
  layout.annotations = [{{ x:0.012, xref:"paper", y:d.baseline, xanchor:"left", yanchor:"bottom",
      text:"GPT-2 XL baseline "+d.baseline.toFixed(3), showarrow:false,
      font:{{size:11, color:"#6b7280"}} }}];
  layout.legend = {{ orientation:"h", y:1.06, x:1, xanchor:"right", font:{{size:11}} }};
  layout.margin.t = 24;
  Plotly.newPlot("plot-"+folder, traces, layout, CFG);
}}

Object.keys(PAYLOAD).forEach(drawRun);

// ---- summary overlays (trimmed & untrimmed shown separately) ----
function summaryPlot(divId, folders, baseline, baseLabel, baseColor) {{
  const traces = [];
  folders.forEach(folder => {{
    const d = PAYLOAD[folder]; const color = PALETTE[folder];
    const nm = d.label + " · " + d.model + " (" + d.effort + ")";
    // faint raw points (legitimate hand-wired iterations only)
    const leg = pick(d, "corr", "legit");
    traces.push({{ x:leg.x, y:leg.y, mode:"markers", showlegend:false,
      marker:{{size:4, color:color, opacity:0.22}}, hoverinfo:"skip", legendgroup:folder }});
    // connected running-best line
    traces.push({{ x:d.it, y:d.rmax, mode:"lines+markers", name:nm, legendgroup:folder,
      line:{{color:color, width:2.5}}, marker:{{size:3, color:color}},
      text:d.name.map((n,i)=>"<b>"+nm+"</b><br>"+n+"<br>running best "+(d.rmax[i]!=null?d.rmax[i].toFixed(4):"—")),
      hoverinfo:"text" }});
  }});
  const layout = JSON.parse(JSON.stringify(BASE_LAYOUT));
  layout.margin.t = 16;
  // extra bottom room + lower legend so the wrapped legend never collides with
  // the "iteration" axis title (the trimmed panel has up to 5 wrapping entries).
  layout.margin.b = 132;
  layout.xaxis = Object.assign({{}}, layout.xaxis, {{ title:{{text:"iteration", standoff:8}} }});
  layout.legend = {{ orientation:"h", y:-0.30, x:0, yanchor:"top", font:{{size:11}} }};
  layout.shapes = [
    {{ type:"line", x0:0, x1:1, xref:"paper", y0:baseline, y1:baseline,
       line:{{color:baseColor, width:1.4, dash:"dash"}} }}
  ];
  layout.annotations = [
    {{ x:0.012, xref:"paper", y:baseline, xanchor:"left", yanchor:"bottom",
       text:baseLabel, showarrow:false, font:{{size:10.5, color:baseColor}} }}
  ];
  Plotly.newPlot(divId, traces, layout, CFG);
}}
// trimmed panel: shared GPT-2 XL baseline (≈0.083) from any trimmed run
summaryPlot("summary-trimmed", TRIMMED, PAYLOAD[TRIMMED[0]].baseline,
  "GPT-2 XL (trimmed) " + PAYLOAD[TRIMMED[0]].baseline.toFixed(3), "#9ca3af");
// untrimmed panel: the single May-27 run against its own (higher) baseline
summaryPlot("summary-untrimmed", [UNTRIMMED], PAYLOAD[UNTRIMMED].baseline,
  "GPT-2 XL (untrimmed) " + PAYLOAD[UNTRIMMED].baseline.toFixed(3), "#d97706");

// Re-fit plots that were hidden inside <details> when they open.
document.querySelectorAll("details.methods-wrap").forEach(el => {{
  el.addEventListener("toggle", () => {{
    window.dispatchEvent(new Event("resize"));
  }});
}});
</script>
</body>
</html>'''

with open(OUT, "w") as f:
    f.write(HTML)
print(f"wrote {OUT} ({len(HTML)} bytes)")
