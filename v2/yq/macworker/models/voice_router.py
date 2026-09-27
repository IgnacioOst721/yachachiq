"""ASR routing decisions (pure logic, no models): which engine hears which language.

Rule (measured, see docs/voice.md):
  - Whisper for languages where Whisper large-v3 is strong (FLEURS WER <= 15 %,
    `Language.whisper_strong`), Omnilingual ASR as the backup.
  - Omnilingual ASR for everything else it supports (Quechua, Aymara, Amazonian
    languages, ~1600 in total), Whisper as a weak backup when it knows the language.
  - "auto": Whisper's language ID for the languages it is strong in, MMS-LID
    (4017 languages) for the rest; when both are unsure, Omnilingual CTC is used
    because it transcribes without being told the language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from yq.common import languages


@dataclass
class Plan:
    lang: str                                   # canonical code, or "" when still unknown
    engines: list = field(default_factory=list)  # ordered model names to try
    reason: str = ""
    candidates: list = field(default_factory=list)   # [[code, prob], ...] from LID


def plan_for_language(lang: str, whisper: str, omni: str, available: Optional[set] = None) -> Plan:
    """Engines for a known language. `whisper`/`omni` = configured model names."""
    avail = available if available is not None else {whisper, omni}
    L = languages.get(lang)
    if L is None:
        return Plan(lang="", engines=[e for e in (whisper,) if e in avail], reason="unknown language: whisper auto")
    w_ok = "whisper" in L.asr and whisper in avail
    o_ok = "omniasr" in L.asr and omni in avail
    if w_ok and L.whisper_strong:
        return Plan(L.code, [whisper] + ([omni] if o_ok else []), "whisper strong for %s" % L.code)
    if o_ok:
        return Plan(L.code, [omni] + ([whisper] if w_ok else []), "omnilingual for %s" % L.code)
    if w_ok:
        return Plan(L.code, [whisper], "only whisper knows %s" % L.code)
    return Plan(L.code, [], "no ASR engine available for %s" % L.code)


def lang_group(code: str) -> str:
    """Dialect continua that LID models split between varieties: sum them before deciding."""
    L = languages.get(code)
    if not L:
        return code
    text = (L.name_en + " " + " ".join(L.aliases)).lower()
    if "quechua" in text or "quichua" in text or "kichwa" in text:
        return "quechua"
    if "aymara" in text:
        return "aymara"
    return L.code


def _mms_top(mms_probs: list):
    """(Language, group probability) for the best group in MMS-LID's answer."""
    mass: dict = {}
    best: dict = {}
    for code, p in mms_probs or []:
        L = languages.get(code)
        if not L:
            continue
        g = lang_group(L.code)
        mass[g] = mass.get(g, 0.0) + float(p)
        if g not in best:                    # list is sorted: first seen = top variety of the group
            best[g] = L
    if not mass:
        return None
    g = max(mass, key=mass.get)
    return best[g], mass[g]


def merge_lid(whisper_probs: list, mms_probs: list) -> list:
    """Combined candidates [[canonical, prob], ...]. Whisper's opinion counts only
    for languages it is strong in; MMS-LID covers the rest."""
    best: dict = {}
    for code, p in whisper_probs or []:
        L = languages.get(code)
        if L and L.whisper_strong:
            best[L.code] = max(best.get(L.code, 0.0), float(p))
    for code, p in mms_probs or []:
        L = languages.get(code)
        if L:
            best[L.code] = max(best.get(L.code, 0.0), float(p))
    return sorted(([c, round(p, 4)] for c, p in best.items()), key=lambda kv: -kv[1])


def plan_auto(whisper_probs: list, mms_probs: list, whisper: str, omni: str,
              whisper_min: float = 0.6, mms_min: float = 0.5, available: Optional[set] = None) -> Plan:
    """Decide language + engines from the two language-ID opinions."""
    avail = available if available is not None else {whisper, omni}
    cands = merge_lid(whisper_probs, mms_probs)
    w_top = None
    for code, p in whisper_probs or []:
        L = languages.get(code)
        if L:
            w_top = (L, float(p))
        break
    m_top = _mms_top(mms_probs)
    # A language Whisper cannot handle well, clearly identified by MMS-LID -> Omnilingual
    if m_top and m_top[1] >= mms_min and not m_top[0].whisper_strong and "omniasr" in m_top[0].asr:
        p = plan_for_language(m_top[0].code, whisper, omni, avail)
        p.reason, p.candidates = "mms-lid %s %.2f" % (m_top[0].code, m_top[1]), cands
        return p
    if w_top and w_top[1] >= whisper_min and w_top[0].whisper_strong:
        p = plan_for_language(w_top[0].code, whisper, omni, avail)
        p.reason, p.candidates = "whisper-lid %s %.2f" % (w_top[0].code, w_top[1]), cands
        return p
    if cands:
        top = languages.get(cands[0][0])
        if top and cands[0][1] >= mms_min:
            p = plan_for_language(top.code, whisper, omni, avail)
            p.reason, p.candidates = "combined lid %s %.2f" % (top.code, cands[0][1]), cands
            return p
    # Unsure: Omnilingual CTC needs no language; else Whisper decides by itself.
    engines = [e for e in (omni, whisper) if e in avail]
    if omni in avail and not omni.startswith("omniasr-ctc"):
        engines = [e for e in (whisper, omni) if e in avail]
    return Plan(lang=cands[0][0] if cands else "", engines=engines, reason="lid unsure", candidates=cands)
