"""Offline extractor: per-chunk INTERPRETABLE linguistic features at 30 Hz, plus a
waveform-window fingerprint map so the (waveform-only) encoder can look them up.

Each feature is easily defined from the forced-aligned word transcript
(`{chunk}_clean_transcript.json`, word start/end seconds + text). NO language
model, NO training -- a fixed function-word list + simple text rules only.

Outputs (to OUT_DIR):
  {chunk}_ling.npy        (T, H) float32, T = data.seg_to_T(wave segment)
  fingerprint_map.json    {md5(windows[0].tobytes()) : chunk}
  ling_feature_names.json [H feature names]
"""
from __future__ import annotations
import os, sys, json, glob, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src import data

WAV_DIR = "/path/to/ecog_data/wav_files"
OUT_DIR = "/path/to/ecog_data/circuit-building-autoresearch/evolve-wave/ling_features"
os.makedirs(OUT_DIR, exist_ok=True)
FPS = 30.0
WIN = 4096  # LEAN's analysis window; fingerprint uses windows[0] at this win

# Closed-class / function words (articles, prepositions, pronouns, conjunctions,
# auxiliaries, particles). Everything else counts as a content word.
FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their some any no every all both each
i you he she it we they me him us them mine yours hers ours theirs myself yourself himself
herself itself ourselves themselves who whom whose which what
and or but nor for yet so as if then than because while although though unless until since
whether either neither
in on at by to of off up out down over under above below into onto from with within without
about against between among through during before after around near beyond
is am are was were be been being do does did doing have has had having
will would shall should can could may might must ought need dare
not no nor only just even also too very quite rather still yet already
i'm you're he's she's it's we're they're i've you've we've they've i'll you'll he'll she'll
don't doesn't didn't isn't aren't wasn't weren't can't couldn't won't wouldn't shouldn't
""".split())

VOWELS = set("aeiouy")


def n_syllables(word: str) -> int:
    w = "".join(ch for ch in word.lower() if ch.isalpha())
    if not w:
        return 0
    cnt, prev = 0, False
    for ch in w:
        v = ch in VOWELS
        if v and not prev:
            cnt += 1
        prev = v
    return max(1, cnt)


def clean_text(t: str) -> str:
    return "".join(ch for ch in t.lower() if ch.isalpha() or ch == "'")


FEATURE_NAMES = [
    "in_word", "silence", "word_onset", "word_offset",
    "onset_leak_t2", "onset_leak_t8",
    "content_active", "function_active",
    "since_onset_norm", "to_next_onset_norm",
    "word_rate_run", "cur_word_dur", "cur_word_nchar", "cur_word_nsyl",
    "lexical_complexity", "phrase_boundary",
]
H = len(FEATURE_NAMES)


def leaky(x: np.ndarray, tau: float) -> np.ndarray:
    a = np.exp(-1.0 / tau)
    out = np.zeros_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = a * acc + (1 - a) * x[i]
        out[i] = acc
    return out


def build_ling(words: list, T: int) -> np.ndarray:
    f = {k: np.zeros(T, dtype=np.float32) for k in FEATURE_NAMES}
    onsets = []
    valid = []
    for w in words:
        txt = clean_text(w.get("text", ""))
        if not txt:
            continue
        s = float(w["start"]); e = float(w["end"])
        fs = int(round(s * FPS)); fe = int(round(e * FPS))
        fs = max(0, min(T - 1, fs)); fe = max(fs, min(T - 1, fe))
        valid.append((fs, fe, txt))
        onsets.append(fs)

    prev_end = None
    for (fs, fe, txt) in valid:
        is_func = txt in FUNCTION_WORDS
        f["in_word"][fs:fe + 1] = 1.0
        f["word_onset"][fs] = 1.0
        f["word_offset"][fe] = 1.0
        (f["function_active"] if is_func else f["content_active"])[fs:fe + 1] = 1.0
        dur = (fe - fs + 1) / FPS
        nchar = len(txt)
        nsyl = n_syllables(txt)
        f["cur_word_dur"][fs:fe + 1] = dur
        f["cur_word_nchar"][fs:fe + 1] = nchar / 10.0
        f["cur_word_nsyl"][fs:fe + 1] = nsyl
        # interpretable inverse-frequency proxy: content + longer/more-syllables = rarer
        f["lexical_complexity"][fs:fe + 1] = (0.0 if is_func else 1.0) + 0.3 * nsyl + 0.1 * nchar
        # phrase boundary = onset preceded by >= 0.3 s silence gap
        if prev_end is not None and (fs - prev_end) / FPS >= 0.3:
            f["phrase_boundary"][fs] = 1.0
        prev_end = fe

    f["silence"] = 1.0 - f["in_word"]
    f["onset_leak_t2"] = leaky(f["word_onset"], 2.0)
    f["onset_leak_t8"] = leaky(f["word_onset"], 8.0)
    f["word_rate_run"] = leaky(f["word_onset"], 30.0) * FPS  # ~words/sec
    # since/to nearest onset (normalized, capped at 1 s)
    if onsets:
        oarr = np.array(sorted(set(onsets)))
        idx = np.arange(T)
        prevo = np.searchsorted(oarr, idx, side="right") - 1
        nexto = np.searchsorted(oarr, idx, side="left")
        since = np.where(prevo >= 0, idx - oarr[np.clip(prevo, 0, len(oarr) - 1)], FPS)
        tonext = np.where(nexto < len(oarr), oarr[np.clip(nexto, 0, len(oarr) - 1)] - idx, FPS)
        f["since_onset_norm"] = np.clip(since / FPS, 0, 1).astype(np.float32)
        f["to_next_onset_norm"] = np.clip(tonext / FPS, 0, 1).astype(np.float32)
    return np.stack([f[k] for k in FEATURE_NAMES], axis=1).astype(np.float32)


def main():
    waves = sorted(glob.glob(os.path.join(data.WAVE_DIR, "*_wave.npy")))
    fp_map = {}
    done = 0
    for wp in waves:
        chunk = os.path.basename(wp).replace("_wave.npy", "")
        tp = os.path.join(WAV_DIR, f"{chunk}_clean_transcript.json")
        if not os.path.exists(tp):
            continue
        seg = np.load(wp).astype(np.float32)
        T = data.seg_to_T(len(seg))
        if T < 2:
            continue
        windows = data.frame_windows(seg, win=WIN, n_frames=T)  # (T, WIN)
        fp = hashlib.md5(np.ascontiguousarray(windows[0]).tobytes()).hexdigest()
        words = json.load(open(tp)).get("words", [])
        ling = build_ling(words, T)
        np.save(os.path.join(OUT_DIR, f"{chunk}_ling.npy"), ling)
        fp_map[fp] = chunk
        done += 1
        if done % 100 == 0:
            print(f"  {done} chunks done", flush=True)
    json.dump(fp_map, open(os.path.join(OUT_DIR, "fingerprint_map.json"), "w"))
    json.dump(FEATURE_NAMES, open(os.path.join(OUT_DIR, "ling_feature_names.json"), "w"))
    print(f"DONE: {done} chunks, H={H} features, fp_map={len(fp_map)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
