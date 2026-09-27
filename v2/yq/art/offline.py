"""Offline story analysis (no language model): story words -> motif keys, title, plan.

Ported from v1 robot/language.py + story.py (the team's own code). Used when the
MacBook is unreachable (procedural drawing) and as the mock planner in tests.
Stems are matched as prefixes on accent-stripped lowercase words (Spanish,
Quechua and English), so "montañas", "montaña", "urqu" all give "mountain".
"""
from __future__ import annotations

import re
import unicodedata

from yq.common.contracts import ScenePlan

ELEMENT_STEMS = [
    # people
    ("niñ", "person"), ("nino", "person"), ("nina", "person"), ("chic", "person"), ("muchach", "person"),
    ("hombre", "person"), ("mujer", "person"), ("abuel", "person"), ("señor", "person"),
    ("senor", "person"), ("pastor", "person"), ("campesin", "person"), ("madre", "person"),
    ("padre", "person"), ("hij", "person"), ("wawa", "person"), ("runa", "person"),
    ("tayta", "person"), ("mama", "person"), ("girl", "person"), ("boy", "person"),
    ("man", "person"), ("woman", "person"), ("child", "person"),
    # animals
    ("condor", "condor"), ("kuntur", "condor"),
    ("llama", "llama"), ("alpaca", "llama"), ("vicuñ", "llama"), ("vicun", "llama"),
    ("paqucha", "llama"), ("wikuñ", "llama"),
    ("puma", "puma"), ("gato", "cat"), ("misi", "cat"), ("cat", "cat"),
    ("perr", "dog"), ("allqu", "dog"), ("dog", "dog"),
    ("pajar", "bird"), ("ave", "bird"), ("pisqu", "bird"), ("bird", "bird"),
    ("colibri", "bird"), ("pez", "fish"), ("pec", "fish"), ("challwa", "fish"),
    ("fish", "fish"), ("trucha", "fish"), ("oso", "bear"), ("ukuku", "bear"),
    ("zorr", "fox"), ("atuq", "fox"), ("serpiente", "snake"), ("culebra", "snake"),
    ("amaru", "snake"), ("snake", "snake"),
    # landscape
    ("montañ", "mountain"), ("montan", "mountain"), ("cerro", "mountain"),
    ("apu", "mountain"), ("urqu", "mountain"), ("nevad", "mountain"), ("andes", "mountain"),
    ("mountain", "mountain"), ("hill", "mountain"),
    ("rio", "river"), ("mayu", "river"), ("river", "river"), ("quebrada", "river"),
    ("lagun", "lake"), ("lago", "lake"), ("qucha", "lake"), ("lake", "lake"),
    ("mar", "sea"), ("playa", "sea"), ("ocean", "sea"), ("sea", "sea"),
    ("sol", "sun"), ("inti", "sun"), ("sun", "sun"),
    ("luna", "moon"), ("killa", "moon"), ("moon", "moon"),
    ("estrella", "star"), ("quyllur", "star"), ("chaska", "star"), ("star", "star"),
    ("nube", "cloud"), ("cloud", "cloud"), ("lluvia", "rain"), ("rain", "rain"),
    ("arbol", "tree"), ("sacha", "tree"), ("tree", "tree"), ("bosque", "tree"),
    ("selva", "tree"), ("flor", "flower"), ("tika", "flower"), ("flower", "flower"),
    ("maiz", "corn"), ("choclo", "corn"), ("sara", "corn"), ("corn", "corn"),
    ("papa", "potato"), ("potato", "potato"), ("chacra", "field"), ("chakra", "field"),
    ("campo", "field"), ("field", "field"),
    # built things
    ("casa", "house"), ("wasi", "house"), ("house", "house"), ("pueblo", "house"),
    ("aldea", "house"), ("village", "house"), ("puente", "bridge"), ("bridge", "bridge"),
    ("camino", "path"), ("sendero", "path"), ("path", "path"), ("road", "path"),
    ("templo", "temple"), ("temple", "temple"), ("iglesia", "temple"),
    ("barco", "boat"), ("bote", "boat"), ("boat", "boat"), ("balsa", "boat"),
    ("fuego", "fire"), ("fire", "fire"),
    ("musica", "music"), ("flauta", "music"), ("quena", "music"), ("zampoña", "music"),
    ("tambor", "music"), ("music", "music"), ("danza", "music"), ("baile", "music"),
]
EXACT_ONLY = {"mar", "sol", "ave", "oso", "pez", "apu", "man", "boy", "cat", "dog"}
ELEMENT_STEMS += [
    ("chakana", "chakana"), ("chacana", "chakana"), ("cruz andina", "chakana"),
    ("anden", "terraces"), ("terraza", "terraces"), ("terrace", "terraces"), ("pata pata", "terraces"),
    ("wayna", "person"), ("sipas", "person"), ("warmi", "person"), ("qhari", "person"),
    ("alpaca", "llama"), ("vicuna", "llama"), ("guanaco", "llama"), ("vizcacha", "fox"),
    ("inti", "sun"), ("killa", "moon"), ("fox", "fox"), ("bear", "bear"), ("hummingbird", "bird"),
]

ENGLISH = {
    "person": "a child in a poncho", "condor": "an Andean condor flying", "llama": "a llama", "puma": "a puma",
    "cat": "a cat", "dog": "a dog", "bird": "a bird", "fish": "a fish", "bear": "a spectacled bear",
    "fox": "an Andean fox", "snake": "a snake", "mountain": "Andean mountains", "river": "a river",
    "lake": "a lake", "sea": "the sea", "sun": "the sun", "moon": "the moon", "star": "stars",
    "cloud": "clouds", "rain": "rain", "tree": "a tree", "flower": "flowers", "corn": "maize plants",
    "potato": "potatoes", "field": "farm fields", "house": "an adobe house", "bridge": "a rope bridge",
    "path": "a mountain path", "temple": "an Inca temple", "boat": "a reed boat", "fire": "a bonfire",
    "music": "a quena flute", "chakana": "a chakana (Andean cross)", "terraces": "Andean farming terraces",
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c) or c == "̃")
    return unicodedata.normalize("NFC", text)


def _words(text: str) -> list:
    return re.findall(r"[a-zñ']+", _norm(text))


def find_elements(text: str) -> list:
    """Ordered motif keys found in the story (v1 algorithm)."""
    found = []
    words = _words(text)
    joined = " ".join(words)
    for w in words:
        for stem, key in ELEMENT_STEMS:
            if " " in stem:
                continue
            if stem in EXACT_ONLY:
                ok = w in (stem, stem + "es", stem + "s")
            else:
                ok = w.startswith(stem.replace("ñ", "n")) or w.startswith(stem)
            if ok and key not in found:
                found.append(key)
                break
    for stem, key in ELEMENT_STEMS:
        if " " in stem and stem in joined and key not in found:
            found.append(key)
    return found


def title_from(text: str, n: int = 5) -> str:
    words = re.findall(r"[\wñáéíóúü']+", text or "")
    t = " ".join(words[:n]).strip()
    return (t[0].upper() + t[1:]) if t else "Una historia"


def plan(text: str, lang: str = "spa_Latn", text_es: str = "", text_en: str = "") -> ScenePlan:
    """A modest but honest ScenePlan built only from words that appear in the story."""
    src = " ".join(x for x in (text, text_es, text_en) if x)
    els = find_elements(src) or ["mountain"]
    subject = ENGLISH.get(els[0], els[0])
    rest = [ENGLISH.get(e, e) for e in els[1:5]]
    scene = subject + (", with " + ", ".join(rest) if rest else "")
    prompt = ("A black ink line drawing on pure white paper: %s. Clean continuous outlines, clear "
              "silhouettes, no shading, no text, no frame." % scene)
    return ScenePlan(title=title_from(text), title_es=title_from(text_es or text),
                     summary_es=(text_es or text)[:200], subject=subject,
                     elements=[ENGLISH.get(e, e) for e in els[:6]], setting="Andes" if "mountain" in els else "",
                     mood="", cultural_notes=[], prompt=prompt, negative="color, shading, text, frame, photo")
