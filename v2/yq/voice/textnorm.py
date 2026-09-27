"""Text normalisation for accuracy measurement (WER/CER) and comparisons.

Lower case, Unicode NFKC, punctuation and symbols removed (apostrophes inside
words are kept: Quechua ejectives like t'ika, ch'aska), whitespace collapsed.
Accents are kept (they matter in Spanish and Quechua spelling). Numbers are not
spelled out, so "3" vs "tres" counts as an error for every engine alike.
"""
from __future__ import annotations

import re
import unicodedata

_APOS = "'’ʼ`´"


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    for a in _APOS[1:]:
        t = t.replace(a, "'")
    out = []
    for ch in t:
        cat = unicodedata.category(ch)
        if ch == "'" or not (cat.startswith("P") or cat.startswith("S")):
            out.append(ch)
        else:
            out.append(" ")
    t = "".join(out)
    t = re.sub(r"(?<![^\W\d_])'|'(?![^\W\d_])", " ", t)   # drop apostrophes not between letters
    return re.sub(r"\s+", " ", t).strip()
