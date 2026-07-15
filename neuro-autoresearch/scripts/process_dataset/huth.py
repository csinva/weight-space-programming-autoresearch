"""Stage the Huth natural-language fMRI dataset (LeBel 2023, OpenNeuro ds003020) into the baked
data/huth/<study>.pt blob -- the dataset used by circuit-building-autoresearch's evolve-neuro.

Faithful to the reference src/features.py + src/eval.py preprocessing: each word's 10-gram is embedded
by the model, the per-word embeddings are Lanczos-downsampled onto the fMRI TR grid, z-scored, and
FIR-delayed (4 delays), then ridge-fit to the TR responses. The MODEL-INDEPENDENT parts are baked
here; the model-dependent Lanczos + z-score + delays run at eval time (see
harness.loaders.huth.apply_feature_pipeline). Per story we bake:
  X = char-id grid of each word's 10-gram (the reference embeds these; here the model IS the embedder)
  Y = the preprocessed BOLD response at each (edge-trimmed) TR    (n_tr, n_voxels)
  L = the Lanczos operator mapping the story's words -> its TRs   (n_tr, n_word)
Alignment mirrors the Huth DataSequence.from_grid EXACTLY: data_times are word MIDPOINTS (start+end)/2,
tr_times are TR MIDPOINTS (index*TR + TR/2), and words are filtered by make_word_ds's bad_words set.
Stories are the reference's EXACT train/test lists (src/data.py), filtered to those staged on disk for
the subject; trimming matches the reference default (responses `data` are already [10:-5]; we drop a
further 30 TRs per end, and Lanczos onto the same TR grid). Runtime reads only data/huth/; ds003020 is
the build reader. The reference uses subjects UTS01/UTS02/UTS03 (the deeply-sampled ones).

Get the raw data once (git-annex/datalad; per subject ~40-47 GB for the full story set):
  datalad clone https://github.com/OpenNeuroDatasets/ds003020
  cd ds003020 && datalad get derivatives/TextGrids \
      derivatives/preprocessed_data/UTS01 derivatives/preprocessed_data/UTS02 derivatives/preprocessed_data/UTS03
Run (once per subject):
  python -m scripts.process_dataset.huth --subject UTS01
"""
from __future__ import annotations

import argparse
import os
import re

import h5py
import numpy as np
import torch

from harness.datasets import PRESETS, with_subject
from harness.loaders.shared import processed_dir
from harness.loaders.timeline import _char_ngram_ids   # the shared 10-gram char encoder (== get_ngrams)

RAW = os.environ.get("HUTH_DS003020", os.path.expanduser("~/Research/neuralset/data/ds003020"))
TR = 2.0                      # ds003020 repetition time (seconds)
SKIP_TRS, EDGE_TRIM = 10, 30  # `data` is trimmed [10:-5]; the reference default trims a further 30/end
NGRAM, MAX_CHARS = 10, 64     # 10-gram char grid; keep last 64 chars (reference SimpleTransformer max_seq_len)
WINDOW = 3                    # Lanczos filter half-width (reference default)
# Words dropped exactly as the reference make_word_ds (bad_words, checked after .strip("{}")).
BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])

# The reference's EXACT story lists (src/data.py). At bake time we filter to those actually staged
# on disk for the subject (ds003020 lacks `onapproachtopluto` + ~13 huge-data-only train stories),
# preserving the reference ordering; train = REF_TRAIN, held-out test = REF_TEST.
REF_TRAIN = [
    "itsabox", "odetostepfather", "inamoment", "hangtime", "ifthishaircouldtalk",
    "goingthelibertyway", "golfclubbing", "thetriangleshirtwaistconnection",
    "igrewupinthewestborobaptistchurch", "tetris", "becomingindian",
    "canplanetearthfeedtenbillionpeoplepart1", "thetiniestbouquet", "swimmingwithastronauts",
    "lifereimagined", "forgettingfear", "stumblinginthedark", "backsideofthestorm", "food",
    "theclosetthatateeverything", "notontheusualtour", "exorcism", "adventuresinsayingyes",
    "thefreedomridersandme", "cocoonoflove", "waitingtogo", "thepostmanalwayscalls",
    "googlingstrangersandkentuckybluegrass", "mayorofthefreaks", "learninghumanityfromdogs",
    "shoppinginchina", "souls", "cautioneating", "comingofageondeathrow",
    "breakingupintheageofgoogle", "gpsformylostidentity", "eyespy", "treasureisland",
    "thesurprisingthingilearnedsailingsoloaroundtheworld", "theadvancedbeginner",
    "goldiethegoldfish", "life", "thumbsup", "seedpotatoesofleningrad", "theshower", "adollshouse",
    "canplanetearthfeedtenbillionpeoplepart2", "sloth", "howtodraw", "quietfire", "metsmagic",
    "penpal", "thecurse", "canadageeseandddp", "thatthingonmyarm", "buck",
    "wildwomenanddancingqueens", "againstthewind", "indianapolis", "alternateithicatom", "bluehope",
    "kiksuya", "afatherscover", "haveyoumethimyet", "firetestforlove",
    "catfishingstrangerstofindmyself", "christmas1940", "tildeath", "lifeanddeathontheoregontrail",
    "vixenandtheussr", "undertheinfluence", "beneaththemushroomcloud", "jugglingandjesus",
    "superheroesjustforeachother", "sweetaspie", "naked", "singlewomanseekingmanwich", "avatar",
    "whenmothersbullyback", "myfathershands", "reachingoutbetweenthebars", "theinterview",
    "stagefright", "legacy", "canplanetearthfeedtenbillionpeoplepart3", "listo",
    "gangstersandcookies", "birthofanation", "mybackseatviewofagreatromance",
    "lawsthatchokecreativity", "threemonths", "whyimustspeakoutaboutclimatechange", "leavingbaghdad",
]
REF_TEST = ["fromboyhoodtofatherhood", "wheretheressmoke", "onapproachtopluto"]


def _staged(subject: str, story: str) -> bool:
    """True only if the story's response content is actually present (a git-annex symlink whose
    content is not fetched is dangling -> os.path.exists is False)."""
    return os.path.exists(os.path.join(RAW, "derivatives/preprocessed_data", subject, f"{story}.hf5"))


def _word_times(textgrid_path: str) -> tuple[list[float], list[str]]:
    """(word_midpoint_seconds, word) for the TextGrid `word` tier -- data_times are word MIDPOINTS
    (start+end)/2, exactly as the Huth DataSequence.from_grid. Parsed directly (the phone tier's
    overlapping intervals trip strict parsers) and handles all three Praat formats in ds003020:
    long "ooTextFile", "chronological", and short "ooTextFile" (exorcism/life/legacy)."""
    txt = open(textgrid_path, encoding="utf-8", errors="ignore").read()
    keep = lambda w: w.lower().strip("{}").strip() not in BAD_WORDS   # exactly make_word_ds's bad_words
    times, words = [], []
    if "chronological" in txt[:80].lower():
        # Interleaved: each interval is "<tier_no> <xmin> <xmax>\n\"text\"". The word tier's 1-based
        # number (its order among the "IntervalTier" declarations) selects its rows.
        widx = re.findall(r'"IntervalTier"\s+"([^"]*)"', txt).index("word") + 1
        for m in re.finditer(r'(?m)^\s*(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*[\r\n]+\s*"([^"]*)"', txt):
            if int(m.group(1)) == widx and keep(m.group(4).strip()):
                times.append((float(m.group(2)) + float(m.group(3))) / 2.0)
                words.append(m.group(4).strip())
    elif re.search(r'name\s*=\s*"word"', txt):
        # Long "ooTextFile": labelled `intervals [N]: xmin = .. xmax = .. text = ".."` under word tier.
        block = next(b for b in re.split(r"item\s*\[\d+\]:", txt) if re.search(r'name\s*=\s*"word"', b))
        for m in re.finditer(r'xmin\s*=\s*([0-9.eE+-]+)\s*xmax\s*=\s*([0-9.eE+-]+)\s*text\s*=\s*"([^"]*)"', block):
            if keep(m.group(3).strip()):
                times.append((float(m.group(1)) + float(m.group(2))) / 2.0)
                words.append(m.group(3).strip())
    else:
        # Short "ooTextFile": bare values -- '"IntervalTier"','"word"',txmin,txmax,n, then n x (xmin,xmax,"text").
        lines = [l.strip() for l in txt.splitlines()]
        for i in range(len(lines) - 1):
            if lines[i] == '"IntervalTier"' and lines[i + 1] == '"word"':
                n, j = int(float(lines[i + 4])), i + 5
                for _ in range(n):
                    w = lines[j + 2].strip('"')
                    if keep(w):
                        times.append((float(lines[j]) + float(lines[j + 1])) / 2.0)
                        words.append(w)
                    j += 3
                break
    return times, words


def _lanczos(newtime: np.ndarray, oldtime: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Lanczos resampling operator (len(newtime) x len(oldtime)); rows = TRs, cols = words. Verbatim
    from the reference features.lanczos_downsample (its sincmat), just vectorised."""
    cutoff = 1.0 / np.mean(np.diff(newtime))
    t = (newtime[:, None] - oldtime[None, :]) * cutoff
    with np.errstate(invalid="ignore", divide="ignore"):
        val = window * np.sin(np.pi * t) * np.sin(np.pi * t / window) / (np.pi ** 2 * t ** 2)
    val[t == 0] = 1.0
    val[np.abs(t) > window] = 0.0
    return val


def _story(subject: str, story: str):
    """(X chars, Y BOLD TRs, L Lanczos) for one story."""
    with h5py.File(os.path.join(RAW, "derivatives/preprocessed_data", subject, f"{story}.hf5")) as f:
        data = np.asarray(f["data"], dtype=np.float32)           # (n_tr0, n_voxels) == full[10:-5]
    Y = data[EDGE_TRIM:-EDGE_TRIM]                                # full[40:-35] (matches the reference)

    times, words = _word_times(os.path.join(RAW, "derivatives/TextGrids", f"{story}.TextGrid"))
    # Lanczos onto the trimmed TR grid, using TR MIDPOINTS (trtimes + TR/2, per DataSequence.from_grid):
    # response row k == full TR (k + SKIP_TRS + EDGE_TRIM), whose midpoint is that index*TR + TR/2.
    newtime = (np.arange(Y.shape[0]) + SKIP_TRS + EDGE_TRIM) * TR + TR / 2.0
    L = _lanczos(newtime, np.asarray(times))                     # (n_tr, n_word)
    X = _char_ngram_ids(words, NGRAM, MAX_CHARS)                 # (n_word, MAX_CHARS)
    return X, torch.from_numpy(Y), torch.from_numpy(L).float()


def build(subject: str) -> None:
    d = with_subject(PRESETS["huth"], subject)
    # Reference's exact lists, filtered to the stories actually staged for this subject.
    train = [s for s in REF_TRAIN if _staged(d.subject, s)]
    test = [s for s in REF_TEST if _staged(d.subject, s)]
    if not train or not test:
        raise RuntimeError(f"{d.subject}: staged {len(train)} train / {len(test)} test stories -- "
                           f"`datalad get derivatives/preprocessed_data/{d.subject}` first.")

    blob = {"ndelays": 4}
    for split, names in (("train", train), ("test", test)):
        Xs, Ys, Ls, wc = [], [], [], []
        for s in names:
            X, Y, L = _story(d.subject, s)
            Xs.append(X)
            Ys.append(Y)
            Ls.append(L)
            wc.append(X.shape[0])
            print(f"  [{split}] {s}: {X.shape[0]} words, {Y.shape[0]} TRs")
        blob[f"X_{split}"] = torch.cat(Xs)
        blob[f"Y_{split}"] = torch.cat(Ys)
        blob[f"{split}_L"] = Ls          # per-story Lanczos operators (applied to per-story model output)
        blob[f"{split}_wc"] = wc         # per-story word counts (to split the model output back per story)

    out = processed_dir("huth")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{d.study}.pt")
    torch.save(blob, path, pickle_protocol=4)
    print(f"[huth] {d.study}: train {len(train)} / test {len(test)} stories; "
          f"X_train {tuple(blob['X_train'].shape)} Y_train {tuple(blob['Y_train'].shape)} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="UTS03", help="e.g. UTS03 (or 3)")
    a = ap.parse_args()
    build(a.subject)
