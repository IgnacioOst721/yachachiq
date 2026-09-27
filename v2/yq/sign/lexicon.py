"""Word completion for fingerspelling: noisy letters -> likely words.

Score of a word w given the letters seen so far (each with its top alternatives + probs):

    score(w) = sum_i log P(letter_i = w_i | what the camera saw)      (observation, beam search)
             + LAMBDA * log P(w)                                       (frequency prior)
             - edit penalties (one missing / extra / swapped letter allowed)

The word may be longer than what was typed (completion). Matching ignores accents (people
fingerspell "CONDOR", we show "cóndor") but keeps ñ.

Lexicons (built by training/sign/build_lexicon.py, shipped in yq/sign/models/):
  lexicon_spa.tsv  Spanish, FrequencyWords 2018 (OpenSubtitles), CC BY-SA 4.0
  lexicon_eng.tsv  English, same source
  story_words.tsv  our own story vocabulary (Spanish, Quechua, English) with a boost
"""
from __future__ import annotations

import bisect
import math
import unicodedata
from pathlib import Path
from typing import Optional

from . import modelstore

LAMBDA = 0.35               # weight of the frequency prior vs the letter evidence
EDIT_COST = 3.0             # nats for one missing/extra letter
DOUBLE_COST = 0.3           # nats for a missed double letter
FLOOR_P = 0.02              # probability given to a letter the classifier did not list
LANG_LEXICONS = {"prl": ("spa", "que"), "ase": ("eng",), "ils": ("eng",)}
ALPHABET = "abcdefghijklmnñopqrstuvwxyz"


def fold(word: str) -> str:
    """Lowercase, strip accents and apostrophes (Quechua ejectives: sach'a -> sacha), keep ñ."""
    out = []
    for ch in word.lower().replace("'", "").replace("\u2019", ""):
        if ch == "ñ":
            out.append("ñ")
            continue
        d = unicodedata.normalize("NFD", ch)
        out.append("".join(c for c in d if unicodedata.category(c) != "Mn"))
    return "".join(out)


class Lexicon:
    def __init__(self, entries: dict):
        """entries: display word -> count (story words get their boost already applied)."""
        total = float(sum(entries.values())) or 1.0
        best: dict = {}
        for w, c in entries.items():
            k = fold(w)
            if not k.isalpha():
                continue
            lp = math.log(c / total)
            if k not in best or lp > best[k][1]:
                best[k] = (w, lp)
        self.keys = sorted(best)
        self.display = {k: best[k][0] for k in self.keys}
        self.logp = {k: best[k][1] for k in self.keys}
        self.min_logp = min(self.logp.values()) if self.logp else -20.0

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, word: str) -> bool:
        return fold(word) in self.logp

    def with_prefix(self, prefix: str, limit: int = 400) -> list:
        i = bisect.bisect_left(self.keys, prefix)
        out = []
        while i < len(self.keys) and self.keys[i].startswith(prefix) and len(out) < limit:
            out.append(self.keys[i])
            i += 1
        return out

    @classmethod
    def load(cls, sign_lang: str) -> "Lexicon":
        entries: dict = {}
        codes = LANG_LEXICONS.get(sign_lang, ("eng",))
        for code in codes:
            p = modelstore.find_model("lexicon_%s.tsv" % code)
            if p:
                for w, c in _read_tsv(p):
                    entries[w] = entries.get(w, 0) + c
        p = modelstore.find_model("story_words.tsv")
        if p:
            top = max(entries.values()) if entries else 1000
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) >= 3 and not line.startswith("#") and parts[0] in codes:
                    w, boost = parts[1].strip(), float(parts[2])
                    entries[w] = max(entries.get(w, 0), int(top * boost))
        return cls(entries)


def _read_tsv(p: Path):
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        w, _, c = line.partition("\t")
        try:
            yield w, int(c)
        except ValueError:
            continue


def _obs_logp(obs: dict, letter: str) -> float:
    return math.log(max(obs.get(letter, 0.0), FLOOR_P))


# Same hand shape, the difference is only the movement (easy to miss): keep both readings alive.
MOTION_TWINS = {"n": "ñ", "ñ": "n", "i": "j", "j": "i"}
TWIN_FACTOR = 0.3


def _with_twins(obs: dict) -> dict:
    out = dict(obs)
    for letter, p in obs.items():
        twin = MOTION_TWINS.get(letter)
        if twin and twin not in out:
            out[twin] = p * TWIN_FACTOR
    return out


def _prefix_beam(observations: list, beam: int = 40) -> list:
    """observations: list of {letter: prob}. -> [(prefix, logp)] best first."""
    hyps = [("", 0.0)]
    for obs in observations:
        obs = _with_twins(obs)
        nxt = []
        for pre, lp in hyps:
            for letter, p in obs.items():
                nxt.append((pre + letter, lp + math.log(max(p, FLOOR_P))))
        nxt.sort(key=lambda h: -h[1])
        hyps = nxt[:beam]
    return hyps


def _edits1(s: str) -> set:
    out = set()
    for i in range(len(s)):
        out.add(s[:i] + s[i + 1:])                                  # extra letter was typed
        if i + 1 < len(s):
            out.add(s[:i] + s[i + 1] + s[i] + s[i + 2:])            # swapped
    return out


def complete(lex: Lexicon, observations: list, n: int = 5, final: bool = False) -> list:
    """Best words for the letters of the CURRENT word.

    observations: [{letter: prob, ...}, ...] one dict per written letter (the committed letter
    plus its alternatives). final=True means the visitor finished the word (no completion:
    prefer words of the same length). Returns [(display word, score 0..1), ...]."""
    if not observations:
        return []
    typed = "".join(max(o, key=o.get) for o in observations)
    cands: dict = {}

    def consider(key: str, obs_lp: float, penalty: float, expected: int = len(observations)):
        extra = len(key) - expected           # letters beyond what the (edited) spelling explains
        if final and extra != 0:
            penalty += EDIT_COST * abs(extra)
        elif extra > 0:
            penalty += 0.15 * extra                      # mild preference for shorter completions
        s = obs_lp + LAMBDA * lex.logp[key] - penalty
        if key not in cands or s > cands[key]:
            cands[key] = s

    beams = _prefix_beam(observations)
    for pre, lp in beams:
        for key in lex.with_prefix(pre, limit=200):
            consider(key, lp, 0.0)
    for pre, lp in beams[:5]:
        for alt in _edits1(pre):
            if not alt:
                continue
            for key in lex.with_prefix(alt, limit=50):
                consider(key, lp, EDIT_COST, len(alt))
        # one letter missing (the visitor skipped it, or it was not committed): insert any letter
        for i in range(1, len(pre) + 1):
            for c in ALPHABET:
                # a missed double letter ("helo" -> "hello") is the most common segmentation miss
                cost = DOUBLE_COST if c == pre[i - 1] else EDIT_COST
                for key in lex.with_prefix(pre[:i] + c + pre[i:], limit=20):
                    consider(key, lp, cost, len(pre) + 1)
    # the literal typed string is always offered (names, words not in the lexicon)
    raw_lp = sum(_obs_logp(o, t) for o, t in zip(observations, typed))
    literal = raw_lp + LAMBDA * lex.min_logp - 1.5 * EDIT_COST
    ranked = sorted(cands.items(), key=lambda kv: -kv[1])[: n * 3]
    out = [(lex.display[k], s) for k, s in ranked]
    if fold(typed) not in cands:
        out.append((typed, literal))
    out.sort(key=lambda kv: -kv[1])
    out = out[:n]
    if not out:
        return []
    m = max(s for _, s in out)
    z = sum(math.exp(s - m) for _, s in out)
    return [(w, math.exp(s - m) / z) for w, s in out]


_CACHE: dict = {}


def get(sign_lang: str) -> Optional[Lexicon]:
    if sign_lang not in _CACHE:
        lex = Lexicon.load(sign_lang)
        _CACHE[sign_lang] = lex if len(lex) else None
    return _CACHE[sign_lang]
