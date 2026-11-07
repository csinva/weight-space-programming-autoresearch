"""
Interpretable transformer embedder for fMRI language encoding.

The agent edits this file. The goal: hand-write the weights of a small
character-level transformer so that the final-token hidden state it produces for
each 10-gram is a good *feature* for predicting fMRI responses to language —
ideally approaching the pretrained GPT-2 XL baseline (`src/baseline.py`).

Rules of the game (same as the sibling `evolve/` project):
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, lookup tables, etc.) inside `write_weights()`.
  * `write_weights()` runs once at construction. It must leave every parameter
    of `SimpleTransformer` populated.
  * Do NOT load pretrained weights and do NOT use external tools to compute the
    embedding — it must come from the transformer forward pass.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --subject UTS03 --num-train 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import (
    EncodingConfig, run_encoding, make_result_row,
    upsert_overall_results, plot_corr_over_iterations,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Experiment knobs. Each iteration changes these plus the shorthand/description.
TOKEN_MODE = 'semantic_only'
POOLING_DECAY = 0.70
TAIL_WINDOW = 192
TAIL_WINDOWS = (8, 24, 64)
TAIL_POOLING = 'mean'
MAX_WORD_CHARS = 16
MAX_WORDS = 11
USE_COMMON_WORD_AXES = True
COMMON_WORD_LIMIT = 6
CUSTOM_COMMON_WORDS = ('the', 'and', 'to', 'of', 'said', 'was', 'had', 'not', 'so', 'think', 'went', 'big', 'all')
SUMMARY_MODE = 'tail_only'
RECENCY_DECAYS = (0.60, 0.80, 0.95)
INCLUDE_TAIL = True
INCLUDE_MEAN = False
INCLUDE_PRESENCE = False
N_RECENCY_BLOCKS = (
    len(RECENCY_DECAYS) if SUMMARY_MODE == "multi_decay"
    else 0 if SUMMARY_MODE in {"tail_only", "multi_tail"}
    else 1
)
N_TAIL_BLOCKS = len(TAIL_WINDOWS) if SUMMARY_MODE == "multi_tail" else int(INCLUDE_TAIL)
N_SUMMARY_BLOCKS = N_RECENCY_BLOCKS + N_TAIL_BLOCKS + int(INCLUDE_MEAN) + int(INCLUDE_PRESENCE)
FEATURE_PROFILE = 'all'
USE_INTERACTIONS = False
USE_POSITIONAL_AXES = False
INCLUDE_BASE_AXES = True
POSITIONAL_DEPTH = 2
USE_POSITIONAL_WORD_AXES = False
WORD_POSITIONAL_DEPTH = 1
USE_HASH_AXES = False
N_HASH_AXES = 16


# Interpretable feature axes. The transformer writes four summaries of these
# same axes into disjoint output blocks: recent exponential mean, tail mean,
# full-context mean, and binary presence.
BASE_FEATURE_NAMES = [
    "word_count", "char_count", "letter_count", "vowel_count", "consonant_count",
    "digit_count", "apostrophe_count", "rare_char", "short_word", "medium_word",
    "long_word", "very_long_word", "suffix_ing", "suffix_ed", "suffix_s",
    "suffix_ly", "first_person", "second_person", "third_person",
    "plural_pronoun", "social_family", "people", "communication", "cognition",
    "emotion_positive", "emotion_negative", "motion", "perception", "body",
    "place_scene", "time", "number_quantity", "object_artifact", "food_drink",
    "animal_nature", "abstract_logic", "negation", "question", "determiner",
    "preposition", "conjunction", "auxiliary", "intensifier", "spatial",
    "auditory", "visual", "manual_action", "possession", "letter_e", "letter_t",
    "letter_a", "letter_o", "letter_i", "letter_n", "letter_s", "letter_h",
    "letter_r", "letter_l", "letter_d", "letter_u", "letter_m", "letter_y",
    "letter_g", "letter_w",
]
DROP_UNUSED_ORTHOGRAPHIC_AXES = True
if DROP_UNUSED_ORTHOGRAPHIC_AXES:
    BASE_FEATURE_NAMES = [
        name for name in BASE_FEATURE_NAMES
        if name not in {
            "char_count", "letter_count", "vowel_count", "consonant_count",
            "digit_count", "apostrophe_count", "rare_char", "letter_e",
            "letter_t", "letter_a", "letter_o", "letter_i", "letter_n",
            "letter_s", "letter_h", "letter_r", "letter_l", "letter_d",
            "letter_u", "letter_m", "letter_y", "letter_g", "letter_w",
        }
    ]
COMMON_WORDS = [
    "the", "and", "to", "of", "a", "in", "that", "it", "is", "was", "i", "he",
    "you", "his", "her", "she", "with", "for", "on", "as", "had", "but", "not",
    "at", "they", "we", "my", "me", "be", "this", "have", "from", "or", "one",
    "all", "so", "there", "what", "said", "out", "up", "when", "about", "who",
    "get", "go", "went", "come", "came", "like", "know", "think", "see", "saw",
    "look", "looked", "tell", "told", "asked", "heard", "time", "day", "night",
    "man", "woman", "boy", "girl", "people", "mother", "father", "friend",
    "house", "home", "room", "door", "car", "street", "hand", "head", "eyes",
    "face", "back", "good", "bad", "little", "big", "old", "new", "first",
    "last", "again", "never", "why", "how", "because", "something", "nothing",
]


def _word_axis(word: str) -> str:
    return "word_" + "".join(ch for ch in word if ch.isalnum())


_ACTIVE_COMMON_WORDS = (
    list(CUSTOM_COMMON_WORDS) if USE_COMMON_WORD_AXES and CUSTOM_COMMON_WORDS
    else COMMON_WORDS[:COMMON_WORD_LIMIT] if USE_COMMON_WORD_AXES
    else []
)
WORD_FEATURES = {word: _word_axis(word) for word in _ACTIVE_COMMON_WORDS}
POSITIONAL_FEATURE_NAMES = [
    f"pos{rel}_{name}"
    for rel in range(POSITIONAL_DEPTH)
    for name in BASE_FEATURE_NAMES
]
HASH_FEATURE_NAMES = [f"lexhash{i}" for i in range(N_HASH_AXES)]
POSITIONAL_WORD_FEATURE_NAMES = [
    f"pos{rel}_{name}"
    for rel in range(WORD_POSITIONAL_DEPTH)
    for name in WORD_FEATURES.values()
]
FEATURE_NAMES = (
    (BASE_FEATURE_NAMES if INCLUDE_BASE_AXES else [])
    + (POSITIONAL_FEATURE_NAMES if USE_POSITIONAL_AXES else [])
    + (HASH_FEATURE_NAMES if USE_HASH_AXES else [])
    + (list(WORD_FEATURES.values()) if USE_COMMON_WORD_AXES else [])
    + (POSITIONAL_WORD_FEATURE_NAMES if USE_POSITIONAL_WORD_AXES else [])
)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
INTERACTION_PAIRS = [
    ("first_person", "cognition"),
    ("first_person", "emotion_negative"),
    ("third_person", "communication"),
    ("social_family", "emotion_positive"),
    ("social_family", "emotion_negative"),
    ("people", "communication"),
    ("people", "motion"),
    ("motion", "place_scene"),
    ("motion", "body"),
    ("perception", "visual"),
    ("perception", "auditory"),
    ("negation", "emotion_negative"),
    ("question", "communication"),
    ("time", "motion"),
    ("possession", "object_artifact"),
    ("body", "emotion_negative"),
    ("manual_action", "object_artifact"),
    ("spatial", "place_scene"),
    ("auxiliary", "motion"),
    ("determiner", "object_artifact"),
]

_VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789'\\"
_VOWELS = set("aeiou")
_COMMON_LETTER_FEATURES = {
    "e": "letter_e", "t": "letter_t", "a": "letter_a", "o": "letter_o",
    "i": "letter_i", "n": "letter_n", "s": "letter_s", "h": "letter_h",
    "r": "letter_r", "l": "letter_l", "d": "letter_d", "u": "letter_u",
    "m": "letter_m", "y": "letter_y", "g": "letter_g", "w": "letter_w",
}


def _feat_token(name: str) -> str:
    return f"<feat:{name}>"


_CHAR_TOKENS = [f"<char:{c}>" for c in _VOCAB_CHARS]
_FEATURE_TOKENS = [_feat_token(name) for name in FEATURE_NAMES]
VOCAB = ["<pad>", "<unk>"] + _CHAR_TOKENS + _FEATURE_TOKENS
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}


SEMANTIC_WORDS: Dict[str, set[str]] = {
    "first_person": {
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    },
    "second_person": {"you", "your", "yours", "yourself", "yourselves"},
    "third_person": {
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
        "its", "itself",
    },
    "plural_pronoun": {"we", "us", "our", "they", "them", "their", "theirs", "everyone", "somebody"},
    "social_family": {
        "mother", "mom", "mommy", "father", "dad", "daddy", "parent", "parents",
        "brother", "sister", "son", "daughter", "child", "children", "kid", "kids",
        "wife", "husband", "boyfriend", "girlfriend", "friend", "friends", "family",
        "grandmother", "grandfather", "grandma", "grandpa",
    },
    "people": {
        "man", "woman", "men", "women", "boy", "girl", "person", "people", "guy",
        "lady", "child", "children", "kid", "teacher", "doctor", "student", "police",
        "cop", "stranger", "neighbor", "boss", "worker", "baby",
    },
    "communication": {
        "say", "said", "says", "tell", "told", "talk", "talked", "speak", "spoke",
        "ask", "asked", "answer", "answered", "call", "called", "shout", "shouted",
        "whisper", "whispered", "voice", "word", "words", "story", "read", "write",
        "heard", "listen", "conversation",
    },
    "cognition": {
        "think", "thought", "know", "knew", "believe", "remember", "forgot", "forget",
        "understand", "wonder", "guess", "decide", "decided", "realize", "realized",
        "idea", "mind", "dream", "imagine", "learn", "learned", "mean", "meant",
    },
    "emotion_positive": {
        "love", "loved", "like", "liked", "happy", "glad", "laugh", "laughed", "smile",
        "smiled", "fun", "funny", "nice", "good", "great", "beautiful", "safe", "hope",
    },
    "emotion_negative": {
        "hate", "hated", "sad", "angry", "mad", "afraid", "scared", "fear", "worried",
        "worry", "cry", "cried", "hurt", "pain", "bad", "wrong", "terrible", "dead",
        "death", "kill", "killed", "alone", "sorry",
    },
    "motion": {
        "go", "goes", "went", "gone", "come", "came", "walk", "walked", "run", "ran",
        "move", "moved", "turn", "turned", "stand", "stood", "sit", "sat", "leave",
        "left", "enter", "entered", "drive", "drove", "fall", "fell", "jump", "climb",
    },
    "perception": {
        "see", "saw", "seen", "look", "looked", "watch", "watched", "hear", "heard",
        "listen", "listened", "feel", "felt", "smell", "taste", "notice", "noticed",
    },
    "body": {
        "hand", "hands", "arm", "arms", "leg", "legs", "head", "face", "eye", "eyes",
        "mouth", "hair", "heart", "body", "back", "feet", "foot", "finger", "fingers",
        "skin", "blood", "brain",
    },
    "place_scene": {
        "room", "house", "home", "street", "road", "car", "school", "city", "town",
        "store", "office", "door", "window", "bed", "table", "kitchen", "bathroom",
        "park", "river", "water", "church", "hospital", "apartment",
    },
    "time": {
        "time", "day", "night", "morning", "evening", "hour", "minute", "second",
        "week", "month", "year", "today", "yesterday", "tomorrow", "then", "now",
        "before", "after", "later", "again", "always", "never",
    },
    "number_quantity": {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "first", "second", "third", "many", "much", "few", "more", "less", "all",
        "some", "any", "each", "every", "both", "half",
    },
    "object_artifact": {
        "thing", "things", "book", "paper", "phone", "letter", "money", "gun", "knife",
        "bag", "box", "chair", "table", "bed", "clothes", "shirt", "shoes", "picture",
        "photo", "key", "keys", "glass", "bottle", "computer",
    },
    "food_drink": {
        "food", "eat", "ate", "eating", "drink", "drank", "water", "coffee", "tea",
        "beer", "wine", "bread", "milk", "meat", "cake", "dinner", "breakfast", "lunch",
    },
    "animal_nature": {
        "dog", "cat", "bird", "horse", "animal", "tree", "trees", "flower", "flowers",
        "sun", "moon", "sky", "rain", "snow", "wind", "fire", "earth", "forest", "sea",
    },
    "abstract_logic": {
        "because", "if", "though", "although", "maybe", "probably", "perhaps", "why",
        "reason", "cause", "truth", "true", "false", "fact", "problem", "question",
    },
    "negation": {"no", "not", "never", "nothing", "nobody", "none", "cannot", "cant", "don't", "didn't"},
    "question": {"who", "what", "when", "where", "why", "how", "which", "question", "ask", "asked"},
    "determiner": {"the", "a", "an", "this", "that", "these", "those", "each", "every"},
    "preposition": {
        "in", "on", "at", "by", "for", "with", "without", "from", "to", "into", "onto",
        "over", "under", "between", "through", "around", "about", "against", "inside",
        "outside", "before", "after",
    },
    "conjunction": {"and", "or", "but", "so", "because", "while", "though", "although", "if", "then"},
    "auxiliary": {
        "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did",
        "have", "has", "had", "will", "would", "can", "could", "should", "may", "might",
        "must",
    },
    "intensifier": {"very", "really", "quite", "too", "so", "just", "almost", "only", "even", "still"},
    "spatial": {
        "up", "down", "left", "right", "front", "back", "inside", "outside", "near",
        "far", "here", "there", "where", "behind", "above", "below",
    },
    "auditory": {"hear", "heard", "sound", "sounds", "listen", "voice", "noise", "music", "song", "loud"},
    "visual": {"see", "saw", "look", "watch", "eye", "eyes", "picture", "dark", "light", "color", "red", "blue"},
    "manual_action": {
        "take", "took", "give", "gave", "hold", "held", "touch", "grab", "pull", "push",
        "open", "opened", "close", "closed", "put", "pick", "picked", "throw", "threw",
    },
    "possession": {"my", "mine", "your", "yours", "his", "her", "hers", "our", "ours", "their", "theirs", "own"},
    "desire_need": {
        "want", "wanted", "wants", "wanting", "need", "needed", "needs", "try",
        "tried", "trying", "hope", "hoped", "wish", "wished", "supposed",
    },
    "memory": {
        "remember", "remembered", "forget", "forgot", "forgotten", "recall",
        "recalled", "memory", "memories", "remind", "reminded",
    },
    "uncertainty": {
        "maybe", "perhaps", "probably", "possibly", "guess", "guessed", "seem",
        "seemed", "wonder", "wondered", "might", "could", "almost", "somehow",
    },
    "danger_violence": {
        "danger", "dangerous", "gun", "knife", "kill", "killed", "hurt", "pain",
        "blood", "dead", "death", "fire", "fight", "fighting", "police", "cop",
        "afraid", "scared", "terrible",
    },
    "work_money": {
        "work", "worked", "worker", "job", "office", "boss", "business", "money",
        "pay", "paid", "buy", "bought", "sell", "sold", "store", "company",
    },
    "school_learning": {
        "school", "class", "teacher", "student", "learn", "learned", "study",
        "studied", "read", "reading", "book", "books", "write", "wrote",
    },
    "transport": {
        "car", "truck", "bus", "train", "plane", "airplane", "boat", "ship",
        "drive", "drove", "driving", "ride", "rode", "road", "street",
    },
    "clothing": {
        "clothes", "shirt", "pants", "dress", "coat", "shoes", "shoe", "hat",
        "jacket", "wear", "wore", "wearing",
    },
    "home_domestic": {
        "home", "house", "room", "kitchen", "bed", "bathroom", "door", "window",
        "table", "chair", "apartment", "floor", "wall",
    },
    "sleep_dream": {
        "sleep", "slept", "sleeping", "dream", "dreamed", "dreamt", "bed",
        "night", "wake", "woke", "awake", "tired",
    },
    "morality_religion": {
        "god", "church", "pray", "prayed", "right", "wrong", "sin", "truth",
        "true", "lie", "lied", "honest", "fair", "blame", "guilt",
    },
    "dialogue_story": {
        "story", "stories", "said", "told", "asked", "answer", "answered",
        "question", "word", "words", "voice", "conversation", "talk", "talked",
    },
}
CONTENT_FEATURES = {
    "social_family", "people", "communication", "cognition", "emotion_positive",
    "emotion_negative", "motion", "perception", "body", "place_scene", "time",
    "number_quantity", "object_artifact", "food_drink", "animal_nature",
    "abstract_logic", "auditory", "visual", "manual_action", "desire_need",
    "memory", "uncertainty", "danger_violence", "work_money",
    "school_learning", "transport", "clothing", "home_domestic",
    "sleep_dream", "morality_religion", "dialogue_story",
}
GRAMMAR_FEATURES = {
    "first_person", "second_person", "third_person", "plural_pronoun", "negation",
    "question", "determiner", "preposition", "conjunction", "auxiliary",
    "intensifier", "spatial", "possession",
}


def _split_words(text: str) -> List[str]:
    cleaned = []
    for ch in text.lower():
        if ch.isalnum() or ch == "'":
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return [w.strip("'") for w in "".join(cleaned).split() if w.strip("'")]


def _word_forms(word: str) -> set[str]:
    forms = {word}
    if word.endswith("'s"):
        forms.add(word[:-2])
    if word.endswith("s") and len(word) > 3:
        forms.add(word[:-1])
    if word.endswith("ed") and len(word) > 4:
        forms.add(word[:-2])
    if word.endswith("ing") and len(word) > 5:
        forms.add(word[:-3])
    return forms


def _word_hash(word: str) -> int:
    h = 2166136261
    for ch in word:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _word_feature_names(word: str) -> List[str]:
    feats = []
    n = len(word)
    if FEATURE_PROFILE in {"all", "structure_only", "content_plus_structure"}:
        feats.append("word_count")
        if n <= 3:
            feats.append("short_word")
        elif n <= 7:
            feats.append("medium_word")
        elif n <= 11:
            feats.append("long_word")
        else:
            feats.append("very_long_word")

        if word.endswith("ing") and n > 4:
            feats.append("suffix_ing")
        if word.endswith("ed") and n > 3:
            feats.append("suffix_ed")
        if word.endswith("s") and n > 3:
            feats.append("suffix_s")
        if word.endswith("ly") and n > 3:
            feats.append("suffix_ly")

    forms = _word_forms(word)
    if USE_HASH_AXES and word:
        feats.append(f"lexhash{_word_hash(word) % N_HASH_AXES}")
    word_feature = WORD_FEATURES.get(word) if USE_COMMON_WORD_AXES else None
    if word_feature is not None and word_feature in FEATURE_INDEX:
        feats.append(word_feature)
    for feature, words in SEMANTIC_WORDS.items():
        if FEATURE_PROFILE == "content_only" and feature not in CONTENT_FEATURES:
            continue
        if FEATURE_PROFILE == "grammar_only" and feature not in GRAMMAR_FEATURES:
            continue
        if FEATURE_PROFILE == "structure_only":
            continue
        if FEATURE_PROFILE == "content_plus_structure" and feature not in CONTENT_FEATURES:
            continue
        if forms & words:
            feats.append(feature)
    return feats


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
    """Deterministic token-feature summarizer with a transformer-shaped API.

    `forward` returns a `(B, T, d_model)` tensor so the fixed embedder contract can
    still read the final-token state. The same summary is broadcast to every token;
    only the last non-pad position is consumed by the evaluation harness.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 192,
        d_model: int = N_SUMMARY_BLOCKS * len(FEATURE_NAMES) + (len(INTERACTION_PAIRS) if USE_INTERACTIONS else 0),
        n_heads: int = 4,
        n_layers: int = 0,
        d_ff: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.n_summary_blocks = N_SUMMARY_BLOCKS
        self.base_dim = min(len(FEATURE_NAMES), d_model // self.n_summary_blocks)
        self.pad_id = 0
        self.pooling_decay = POOLING_DECAY
        self.recency_decays = RECENCY_DECAYS
        self.tail_window = TAIL_WINDOW

        self.token_emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        base = self.token_emb(ids)[..., :self.base_dim]
        mask = ids.ne(self.pad_id)
        mask_f = mask.to(base.dtype)
        lengths = mask_f.sum(dim=1).clamp_min(1.0)

        mean = (base * mask_f[..., None]).sum(dim=1) / lengths[:, None]
        presence = torch.where(mask[..., None], base, torch.zeros_like(base)).amax(dim=1)

        pos = torch.arange(T, device=ids.device, dtype=base.dtype)[None, :]
        distance = (lengths[:, None] - 1.0 - pos).clamp_min(0.0)
        recents = []
        if SUMMARY_MODE == "multi_decay":
            decays = self.recency_decays
        elif SUMMARY_MODE == "tail_only":
            decays = ()
        else:
            decays = (self.pooling_decay,)
        for decay_value in decays:
            decay = torch.as_tensor(decay_value, device=ids.device, dtype=base.dtype)
            recent_w = torch.pow(decay, distance) * mask_f
            recents.append(
                (base * recent_w[..., None]).sum(dim=1)
                / recent_w.sum(dim=1).clamp_min(1e-6)[:, None]
            )

        tails = []
        tail_windows = TAIL_WINDOWS if SUMMARY_MODE == "multi_tail" else (self.tail_window,)
        for window in tail_windows:
            tail_start = lengths[:, None] - float(window)
            tail_mask = (pos >= tail_start) & mask
            tail_f = tail_mask.to(base.dtype)
            tail_sum = (base * tail_f[..., None]).sum(dim=1)
            if TAIL_POOLING == "sum":
                tails.append(tail_sum)
            elif TAIL_POOLING == "word_mean" and "word_count" in FEATURE_INDEX and FEATURE_INDEX["word_count"] < tail_sum.shape[1]:
                denom = tail_sum[:, FEATURE_INDEX["word_count"]].clamp_min(1.0)
                tails.append(tail_sum / denom[:, None])
            else:
                tails.append(tail_sum / tail_f.sum(dim=1).clamp_min(1.0)[:, None])

        parts = list(recents)
        if INCLUDE_TAIL:
            parts.extend(tails)
        if INCLUDE_MEAN:
            parts.append(mean)
        if INCLUDE_PRESENCE:
            parts.append(presence)

        interactions = []
        if USE_INTERACTIONS:
            for left, right in INTERACTION_PAIRS:
                li = FEATURE_INDEX[left]
                ri = FEATURE_INDEX[right]
                if li < self.base_dim and ri < self.base_dim:
                    interactions.append((presence[:, li] * presence[:, ri])[:, None])
        summary = torch.cat(parts + interactions, dim=1)
        if summary.shape[1] < self.d_model:
            summary = F.pad(summary, (0, self.d_model - summary.shape[1]))
        elif summary.shape[1] > self.d_model:
            summary = summary[:, :self.d_model]
        return summary[:, None, :].expand(B, T, self.d_model).contiguous()


class InterpretableEmbedder:
    """Tokenizes each string into characters, runs `SimpleTransformer`, and returns
    the hidden state of the final (non-pad) token. Exposes the embedder interface
    `__call__(texts) -> np.ndarray (n_texts, d_model)` used by the encoding pipeline."""

    def __init__(self, model: SimpleTransformer, device: str = 'cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.stoi = {c: i for i, c in enumerate(VOCAB)}
        self.pad_id = 0
        self.unk_id = 1
        self.max_seq_len = model.max_seq_len

    def encode(self, text: str) -> List[int]:
        words = _split_words(text)[-MAX_WORDS:]
        ids: List[int] = []
        for i, word in enumerate(words):
            if TOKEN_MODE != "char_only":
                rel = len(words) - 1 - i
                for name in _word_feature_names(word):
                    if INCLUDE_BASE_AXES and name in FEATURE_INDEX:
                        ids.append(self.stoi[_feat_token(name)])
                    if USE_POSITIONAL_AXES and rel < POSITIONAL_DEPTH and name in BASE_FEATURE_NAMES:
                        pos_name = f"pos{rel}_{name}"
                        if pos_name in FEATURE_INDEX:
                            ids.append(self.stoi[_feat_token(pos_name)])
                    if name in WORD_FEATURES.values() and name in FEATURE_INDEX:
                        ids.append(self.stoi[_feat_token(name)])
                    if USE_POSITIONAL_WORD_AXES and rel < WORD_POSITIONAL_DEPTH and name in WORD_FEATURES.values():
                        pos_name = f"pos{rel}_{name}"
                        if pos_name in FEATURE_INDEX:
                            ids.append(self.stoi[_feat_token(pos_name)])
            if TOKEN_MODE != "semantic_only":
                for ch in word[-MAX_WORD_CHARS:]:
                    ids.append(self.stoi.get(f"<char:{ch}>", self.unk_id))
        ids = ids[-self.max_seq_len:]  # keep the most recent chars (final token matters)
        return ids if ids else [self.pad_id]

    @torch.no_grad()
    def __call__(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            enc = [self.encode(t) for t in texts[i: i + batch_size]]
            lens = [len(e) for e in enc]
            T = max(lens)
            ids = torch.full((len(enc), T), self.pad_id, dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e, dtype=torch.long)
            ids = ids.to(self.device)
            hidden = self.model(ids)  # (B, T, d_model)
            last = torch.tensor([l - 1 for l in lens], device=self.device)
            emb = hidden[torch.arange(len(enc), device=self.device), last]
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer) -> None:
    """Populate `model`'s parameters in-place. No training allowed.

    Each token row is a small hand-written feature vector. The forward pass pools
    those rows in transparent recency/context summaries.
    """
    def add_feature(weight: torch.Tensor, token: str, feature: str, value: float = 1.0) -> None:
        if token not in TOKEN_TO_ID or feature not in FEATURE_INDEX:
            return
        idx = FEATURE_INDEX[feature]
        if idx < model.base_dim:
            weight[TOKEN_TO_ID[token], idx] += value

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        weight = model.token_emb.weight

        add_feature(weight, "<unk>", "rare_char")
        add_feature(weight, "<unk>", "char_count")

        for feature in FEATURE_NAMES:
            add_feature(weight, _feat_token(feature), feature)

        for ch in _VOCAB_CHARS:
            token = f"<char:{ch}>"
            add_feature(weight, token, "char_count")
            if ch.isalpha():
                add_feature(weight, token, "letter_count")
                if ch in _VOWELS:
                    add_feature(weight, token, "vowel_count")
                else:
                    add_feature(weight, token, "consonant_count")
                letter_feature = _COMMON_LETTER_FEATURES.get(ch)
                if letter_feature is not None:
                    add_feature(weight, token, letter_feature)
            elif ch.isdigit():
                add_feature(weight, token, "digit_count")
                add_feature(weight, token, "number_quantity", 0.5)
            elif ch == "'":
                add_feature(weight, token, "apostrophe_count")
            else:
                add_feature(weight, token, "rare_char")
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = 'semantic_bestlex_pair_big_all_ctx11_tailall_v3042'
model_description = 'Never-stop compact semantic/exact-word model with exact axes for big and all (ctx11_tailall).'


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def build_embedder(device: str = 'cuda',
                   d_model: int = N_SUMMARY_BLOCKS * len(FEATURE_NAMES) + (len(INTERACTION_PAIRS) if USE_INTERACTIONS else 0),
                   n_heads: int = 4, n_layers: int = 0,
                   d_ff: int = 0, max_seq_len: int = 192) -> InterpretableEmbedder:
    model = SimpleTransformer(
        vocab_size=len(VOCAB), max_seq_len=max_seq_len,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    write_weights(model)
    model.eval()
    return InterpretableEmbedder(model, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="UTS03")
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    t0 = time.time()
    cfg = EncodingConfig(subject=args.subject, num_train=args.num_train, num_test=args.num_test)
    embedder = build_embedder(device=args.device)
    r = run_encoding(embedder, cfg)
    n_params = sum(p.numel() for p in embedder.model.parameters())

    upsert_overall_results(
        [make_result_row(r, model_shorthand_name, n_params, model_description)], RESULTS_DIR)
    plot_corr_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"subject:        {cfg.subject}")
    print(f"test_corr:      {r['test_corr']:.4f}  (train_corr={r['corrs_train_mean']:.4f}, "
          f"median={r['corrs_test_median']:.4f}, frac>0.2={r['corrs_test_frac>0.2']:.4f}, "
          f"top5%={r['corrs_test_mean_top5_percentile']:.4f})")
    print(f"roi corrs:      " + ", ".join(f"{k}={v:.3f}" for k, v in r['roi_corrs'].items()))
    print(f"encoding_secs:  {r['encoding_seconds']:.1f}s")
    print(f"total_seconds:  {time.time() - t0:.1f}s")
