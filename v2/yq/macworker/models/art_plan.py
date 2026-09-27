"""Story -> ScenePlan with the LLM (faithful, culturally accurate, drawable by a pen).

    clean(text, lang) -> str                        spelling/punctuation only, meaning unchanged
    plan(text, lang, text_es="", text_en="") -> (ScenePlan, text_es, text_en)
"""
from __future__ import annotations

import difflib
import logging
import re

from yq.common import config
from yq.common.contracts import ScenePlan

from . import art_glossary, art_style

log = logging.getLogger("yq.art.plan")

PLAN_SYSTEM = """You are the art director of Yachachiq, a drawing robot built by students in Lima, Peru.
A visitor told a short story (in any language: often Spanish or Quechua, sometimes English or others).
You plan ONE illustration that a pen plotter will draw in black ink on white A4 paper (portrait).

Rules:
1. FAITHFUL: draw who, what and where the story says. Never add characters, animals or objects that
   the story does not mention or clearly imply. Pick the single most important moment. At most 5 main things.
2. CULTURALLY ACCURATE, no stereotypes. If the story is Andean (Quechua words, Peru, Bolivia, the Andes),
   use correct Andean iconography only when the story calls for it:
   - Andean condor: huge black bird, white collar ruff around the neck, bare head (male has a comb), long
     fingered wing tips. Llama: tall, long neck, banana-shaped ears, carries loads. Alpaca: smaller, fluffy,
     short straight ears. Vicuña: slender, wild, tan with a white chest. Vizcacha: rabbit-like rodent with a
     long curled tail living among rocks. Puma: big plain cat, long tail. Spectacled bear (ukuku).
   - Plants and food: maize (sara), potatoes (papa), quinoa, coca leaves, ichu grass, queñua trees.
   - Places: apus (sacred snowy mountains), stone andenes (farming terraces), adobe houses with straw roofs,
     Inca stone walls, Lake Titicaca reed boats (totora), rope bridges (q'eswachaka).
   - Clothing: chullo (knitted hat with ear flaps), poncho, lliclla (woven shawl), montera, pollera, ojotas.
   - Symbols: chakana (stepped cross), inti (sun), killa (moon), Andean textile patterns.
   NEVER use Mexican sombreros, Aztec or Maya pyramids, cactus deserts, feather headdresses or caricatures
   for Andean stories. For stories from other cultures, respect THAT culture with the same care.
3. DRAWABLE: one clear scene, main subject large near the centre, clear silhouettes, little overlap,
   simple background (a few lines of mountains or horizon at most). It must read well at A4 size.
4. "elements": 2-5 short concrete English noun phrases that MUST be visible and can be checked by looking
   at the picture (e.g. "condor", "llama", "girl wearing a chullo", "snowy mountain"). No abstract ideas,
   no "sky", "ground", "background" or "landscape": only the characters and objects of the story.
5. "scene": one English description for an image generator, 25-60 words, subject and action first, then
   setting. Describe shapes and poses, not feelings. Do not mention colours, style, pencil or ink.

Reply with JSON only:
{"title": "short title in the story's own language",
 "title_es": "the title in Spanish",
 "summary_es": "one or two simple Spanish sentences retelling the story",
 "text_en": "faithful English translation of the whole story",
 "text_es": "faithful Spanish translation of the whole story (copy it if it is already Spanish)",
 "culture": "andean | <other culture name> | none",
 "subject": "main subject in English",
 "elements": ["...", "..."],
 "setting": "where, in English",
 "mood": "one or two words",
 "cultural_notes": ["short English notes on the iconography you chose and why"],
 "scene": "..."}"""

PLAN_SCHEMA = {"title": str, "title_es": str, "summary_es": str, "subject": str, "elements": list, "scene": str}

CLEAN_SYSTEM = ("You correct transcripts of stories told aloud to a robot. Fix only spelling, accents, "
                "capital letters and punctuation. Do NOT change, add, remove or reorder words or ideas, do not "
                "translate, do not summarise. Keep Quechua, Aymara and other words exactly as they are. "
                "Reply with the corrected text only.")


def _similar(a: str, b: str) -> float:
    wa = re.findall(r"\w+", a.lower())
    wb = re.findall(r"\w+", b.lower())
    return difflib.SequenceMatcher(None, wa, wb).ratio()


def clean(text: str, lang: str = "spa_Latn") -> str:
    """Light correction; falls back to the original when the model changed the meaning."""
    from . import llm
    t = " ".join((text or "").split())
    if not t or config.mock("llm"):
        return (t[:1].upper() + t[1:]) if t else t
    out = llm.chat([{"role": "system", "content": CLEAN_SYSTEM},
                    {"role": "user", "content": "Language: %s\n\n%s" % (lang, t)}],
                   max_tokens=max(200, int(len(t) / 2.5)), temperature=0.0)
    out = " ".join(out.strip().strip('"').split())
    if not out or _similar(t, out) < 0.6 or not (0.7 < len(out) / max(1, len(t)) < 1.4):
        log.info("clean rejected (changed too much)")
        return t
    return out


def _user_message(text: str, lang: str, text_es: str, text_en: str) -> str:
    parts = ["Story language code: %s" % lang, "Story as told:\n%s" % text.strip()]
    if text_es and text_es.strip() != text.strip():
        parts.append("Spanish translation (machine, may be imperfect):\n%s" % text_es.strip())
    if text_en and text_en.strip() != text.strip():
        parts.append("English translation (machine, may be imperfect):\n%s" % text_en.strip())
    return "\n\n".join(parts)


def _clean_list(v, limit: int) -> list:
    if isinstance(v, str):
        v = [x.strip() for x in re.split(r"[;,]", v)]
    out = []
    for x in v or []:
        x = " ".join(str(x).split()).strip(" .")
        if x and x.lower() not in [o.lower() for o in out]:
            out.append(x)
    return out[:limit]


def _translate(text: str, src: str, tgt: str) -> str:
    try:
        from . import voice_translate                     # owned by VOICE; optional here
        out = voice_translate.translate(text, src, tgt)
        return out if isinstance(out, str) and out.strip() and out.strip() != text.strip() else ""
    except Exception as e:                                # not installed / model missing: the LLM translates
        log.info("voice_translate unavailable (%s)", type(e).__name__)
        return ""


def _salvage(text: str) -> dict:
    """Pull the fields out of an almost-JSON reply (unescaped quote, missing comma, cut off)."""
    d: dict = {}
    for key in ("title", "title_es", "summary_es", "text_en", "text_es", "culture", "subject", "setting",
                "mood", "scene"):
        m = re.search(r'"%s"\s*:\s*"(.*?)"[ \t]*(?:,|\n|\}|$)' % key, text, flags=re.S)
        if m:
            d[key] = m.group(1).replace('\\"', '"').strip()
    for key in ("elements", "cultural_notes"):
        m = re.search(r'"%s"\s*:\s*\[(.*?)\]' % key, text, flags=re.S)
        if m:
            d[key] = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    return d


TITLE_SYSTEM = ("Escribe un título corto (de 2 a 6 palabras) en español para esta historia contada a un robot. "
                "Nombra al protagonista o lo que pasa. Responde solo con el título, sin comillas ni punto final.")


def good_title(t: str, story: str = "") -> bool:
    """2..10 words, or one word that is not just the story's last word copied (the LLM's classic slip)."""
    words = re.findall(r"[\w'’]+", t or "")
    if not words or len(words) > 10 or len(" ".join(words)) > 70:
        return False
    if len(words) == 1:
        story_words = re.findall(r"[\w'’]+", (story or "").lower())
        if story_words and words[0].lower() == story_words[-1]:
            return False
    return True


def _spanish_title(text_es: str) -> str:
    from yq.art import offline
    from . import llm
    if text_es.strip():
        try:
            t = llm.chat([{"role": "system", "content": TITLE_SYSTEM}, {"role": "user", "content": text_es.strip()}],
                         max_tokens=30, temperature=0.2)
            t = " ".join(t.strip().strip('"«»“”.').split())
            if good_title(t, text_es):
                return t[:1].upper() + t[1:]
        except Exception as e:                            # the title must never stop the drawing
            log.info("spanish title failed: %s", e)
    return offline.title_from(text_es) if text_es.strip() else "Una historia"


def fix_titles(title: str, title_es: str, text: str, lang: str, text_es: str) -> tuple:
    """(title in the story's language, title in Spanish), both checked."""
    title, title_es = " ".join((title or "").split()), " ".join((title_es or "").split())
    if lang.startswith("spa"):
        t = title if good_title(title, text) else (title_es if good_title(title_es, text) else _spanish_title(text))
        return t, t
    untranslated = title_es and title and title_es.lower() == title.lower()
    if untranslated or not good_title(title_es, text_es or text):
        title_es = _spanish_title(text_es or text)
    if not good_title(title, text):
        title = _translate(title_es, "spa_Latn", lang) or title_es
    return title, title_es


def plan(text: str, lang: str = "spa_Latn", text_es: str = "", text_en: str = "", backend: str = None):
    """Returns (ScenePlan, text_es, text_en)."""
    from yq.art import offline
    from . import llm
    if config.mock("llm"):
        p = offline.plan(text, lang, text_es, text_en)
        p.prompt = art_style.compose_prompt(p.subject + (", " + ", ".join(p.elements[1:]) if p.elements[1:] else ""),
                                            backend)
        return p, text_es or (text if lang.startswith("spa") else ""), text_en
    # the voice domain's translator (NLLB/MADLAD, CONTRACTS §3.1) handles Quechua far better than an
    # 8B LLM: when the UI did not send translations, get them first and give them to the planner
    if not lang.startswith(("spa", "eng")) and not text_es:
        text_es = _translate(text, lang, "spa_Latn")
    msgs = [{"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": _user_message(text, lang, text_es, text_en)}]
    try:
        d = llm.chat_json(msgs, schema=PLAN_SCHEMA, max_tokens=1600, temperature=0.2)
    except ValueError as e:
        d = _salvage(getattr(e, "last_text", "") or "")
        if not d.get("scene") or not d.get("elements"):
            raise
        log.warning("plan JSON salvaged field by field")
    elements = art_glossary.drawable_elements(_clean_list(d.get("elements"), 5) or
                                              [str(d.get("subject") or "landscape")])
    scene = " ".join(str(d.get("scene") or d.get("subject") or "").split())
    out_es = text_es or (text if lang.startswith("spa") else str(d.get("text_es") or ""))
    out_en = text_en or (text if lang.startswith("eng") else str(d.get("text_en") or ""))
    notes = _clean_list(d.get("cultural_notes"), 6)
    culture = str(d.get("culture") or "").strip()
    if culture and culture.lower() != "none":
        notes = ["culture: " + culture] + notes
    # exact looks the generator gets wrong (condor = vulture with bald head, Andean adobe houses...)
    scene = art_glossary.enrich_scene(scene, elements, culture)
    title, title_es = fix_titles(str(d.get("title") or ""), str(d.get("title_es") or ""), text, lang, out_es)
    p = ScenePlan(title=title or offline.title_from(text),
                  title_es=title_es or title or offline.title_from(out_es or text),
                  summary_es=str(d.get("summary_es") or "").strip(),
                  subject=str(d.get("subject") or elements[0]).strip(),
                  elements=elements, setting=str(d.get("setting") or "").strip(),
                  mood=str(d.get("mood") or "").strip(), cultural_notes=notes,
                  prompt=art_style.compose_prompt(scene, backend), negative=art_style.negative(backend))
    return p, out_es, out_en
