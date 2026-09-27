"""Andean terms the translators get wrong, and a guard against garbage translations.

Found on 2026-09-27 translating one story through the worker:
  spa->eng  "cuida a las llamas"  -> "watches over the flames"        (llama = flame in Spanish)
  spa->quy  "las llamas"          -> "nina ruphay..." (fire)
  quy->spa  "kunturmi karqan"     -> "había un águila"                (kuntur = condor)
  spa->agr  (MADLAD)              -> "13:20 Tuja aishnuma, tuja aishnuma..." (an invented Bible verse)

    postedit(src_text, src, out_text, tgt) -> out_text with the wrong renderings fixed
    degenerate(src_text, out_text)         -> True when the output is garbage (verse numbers, loops...)
    proxy(code)                            -> NLLB stand-in for close varieties (Cusco Quechua -> quy_Latn)
"""
from __future__ import annotations

import re
from typing import Dict, List

# Southern Quechua (Quechua IIC) varieties are close to Ayacucho Quechua, which NLLB supports; without a
# proxy they fall back to MADLAD-400, which invents text for these languages.
PROXY = {"quz_Latn": "quy_Latn", "qxp_Latn": "quy_Latn", "qve_Latn": "quy_Latn", "quh_Latn": "quy_Latn",
         "qul_Latn": "quy_Latn"}


def proxy(code: str) -> str:
    return PROXY.get(code, code)


def family(code: str) -> str:
    c = (code or "").split("_")[0]
    if c in ("spa", "eng"):
        return c
    if c in ("ayr", "ayc", "aym"):
        return "aym"
    if c.startswith("qu") or c.startswith(("qv", "qw", "qx")):
        return "que"
    return c


# concept -> forms per family (lower case), plus renderings that are WRONG in a target family
TERMS: List[Dict] = [
    {"id": "condor", "forms": {"spa": ["cóndor", "condor", "cóndores"], "que": ["kuntur"], "aym": ["kunturi", "kuntur"],
                               "eng": ["condor", "condors"]},
     "wrong": {"spa": {"águila": "cóndor", "águilas": "cóndores", "halcón": "cóndor"},
               "eng": {"eagle": "condor", "eagles": "condors", "hawk": "condor"},
               "que": {"anka": "kuntur"}}},
    {"id": "llama", "forms": {"spa": ["llama", "llamas"], "que": ["llama"], "aym": ["qarwa", "llama"],
                              "eng": ["llama", "llamas"]},
     "wrong": {"eng": {"flame": "llama", "flames": "llamas"}, "que": {"nina": "llama"}, "aym": {"nina": "qarwa"}},
     # "llama" is also "flame" and the verb "se llama": only fix when the story is about the animal
     "skip_if": r"\b(fuego|fogata|fogón|incendio|quem|arde|vela|antorcha|fire|burn|candle|torch)",
     "src_pattern": {"spa": r"(?<!\bse )(?<!\bme )(?<!\bte )(?<!\ble )(?<!\bnos )\bllamas?\b"}},
    {"id": "fox", "forms": {"spa": ["zorro", "zorros"], "que": ["atuq", "atoq"], "aym": ["qamaqi"], "eng": ["fox"]},
     "wrong": {}},
    {"id": "potato", "forms": {"spa": ["papa", "papas"], "que": ["papa"], "eng": ["potato", "potatoes"]},
     "wrong": {"eng": {"pope": "potato", "popes": "potatoes"}},
     "skip_if": r"\b(vaticano|iglesia|santo padre|pope|church)"},
    {"id": "vicuna", "forms": {"spa": ["vicuña", "vicuñas"], "que": ["wik'uña", "wikuña"], "aym": ["wari"],
                               "eng": ["vicuña", "vicuna"]}, "wrong": {}},
]


def _has(text: str, forms: List[str], pattern: str = "") -> bool:
    t = (text or "").lower()
    if pattern:
        return re.search(pattern, t) is not None
    return any(re.search(r"\b%s" % re.escape(f), t) for f in forms)


def _replace_word(text: str, wrong: str, right: str) -> str:
    """Replace a wrong word (Quechua/Aymara: also inside agglutinated words, e.g. nina-kuna-wan)."""
    def fix(m):
        w = m.group(0)
        out = right + w[len(wrong):]
        return out[:1].upper() + out[1:] if w[:1].isupper() else out
    return re.sub(r"\b%s\w*" % re.escape(wrong), fix, text, flags=re.I) if wrong in ("nina", "anka") \
        else re.sub(r"\b%s\b" % re.escape(wrong), lambda m: right.capitalize() if m.group(0)[:1].isupper() else right,
                    text, flags=re.I)


def postedit(src_text: str, src: str, out_text: str, tgt: str) -> str:
    sf, tf = family(src), family(tgt)
    out = out_text
    for term in TERMS:
        if not _has(src_text, term["forms"].get(sf, []), term.get("src_pattern", {}).get(sf, "")):
            continue
        skip = term.get("skip_if")
        if skip and re.search(skip, (src_text or "").lower()):
            continue
        right_forms = term["forms"].get(tf, [])
        for wrong, right in term.get("wrong", {}).get(tf, {}).items():
            if re.search(r"\b%s" % re.escape(wrong), out, flags=re.I):
                if right_forms and _has(out, right_forms) and wrong not in ("nina", "anka"):
                    continue        # both present: the wrong word may be meant (a real eagle and a condor)
                out = _replace_word(out, wrong, right)
    return out


_VERSE = re.compile(r"(^|\s)\d{1,3}:\d{1,3}(\s|$)")


def degenerate(src_text: str, out_text: str) -> bool:
    """Garbage detector: verse numbers the source doesn't have, a phrase looping, absurd length."""
    out = (out_text or "").strip()
    if not out:
        return True
    if _VERSE.search(out) and not _VERSE.search(src_text or ""):
        return True
    words = re.findall(r"\w+", out.lower())
    if len(words) >= 6:
        grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        if max(grams.count(g) for g in set(grams)) >= 3:
            return True
    ratio = len(out) / max(1, len((src_text or "").strip()))
    return not (0.25 <= ratio <= 4.0)
