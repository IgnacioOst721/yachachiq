"""Andean visual glossary: exact looks for subjects image models get wrong.

The generators know "eagle" far better than "Andean condor" and draw European villages
for "village". The planner already describes these subjects, but its one-sentence scene
loses the details, so they are re-added here deterministically:

    enrich_scene(scene, elements, culture)  -> scene + precise descriptions (for the generator)
    verify_hint(element, culture)           -> what the checker must see (for the VLM)

Only entries that match the story are used; entries marked andean_only need the plan's
culture to be Andean (a lighthouse village in an English story keeps its own houses).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# (pattern, generator description, verification hint, andean_only)
GLOSSARY: List[Tuple[str, str, str, bool]] = [
    (r"\bcondors?\b|\bkuntur",
     "the Andean condor is a huge vulture, not an eagle: completely bald wrinkled head and bare neck, "
     "a fluffy white ring of feathers around the base of the neck, short heavy hooked beak, very broad "
     "long wings with long separated finger-like feathers at the wing tips",
     "an Andean condor: bald head and a white ring of feathers around the neck (an eagle with a "
     "feathered head does NOT count)", False),
    (r"\bllamas?\b",
     "the llama has a long upright neck, a long face, tall curved banana-shaped ears and a woolly body "
     "on long thin legs",
     "a llama (long upright neck, tall curved ears)", False),
    (r"\balpacas?\b",
     "the alpaca is smaller and fluffier than a llama, with short straight pointed ears and a fluffy "
     "tuft of wool on top of the head",
     "an alpaca (small, fluffy, short pointed ears)", False),
    (r"\bvicu[ñn]as?\b",
     "the vicuña is a slender wild camelid with a long thin neck, a white chest bib and a light build",
     "a vicuña (slender camelid)", False),
    (r"\bvizcachas?\b",
     "the vizcacha looks like a rabbit with long ears and a long curled bushy tail, sitting on rocks",
     "a vizcacha (rabbit-like, long curled tail)", False),
    (r"\bpumas?\b",
     "the puma is a large plain cat without spots or mane, with a long thick tail",
     "a puma (large plain cat, long tail)", False),
    (r"\b(spectacled bears?|andean bears?|ukuku)\b",
     "the spectacled bear is a stocky black bear with light rings around its eyes like glasses",
     "a spectacled bear", False),
    (r"\b(snowy|snow-capped|snowcapped|snow)\b.*\b(mountains?|peaks?)\b|\bapus?\b|\bnevados?\b",
     "the mountains are jagged Andean peaks with snow caps drawn as clear outlined patches near the tops",
     "mountains with snow on their peaks", False),
    (r"\b(village|houses?|huts?|homes?|town)\b",
     "the houses are small Andean adobe houses with steep thatched straw roofs and stone walls, "
     "with no pine trees and no chimneys",
     "small houses", True),
    (r"\bterraces?\b|\bandenes\b",
     "stone farming terraces (andenes) step down the mountainside like a giant staircase",
     "stone farming terraces on a mountainside", True),
    (r"\bchullos?\b|\bknitted hats?\b",
     "a chullo is a knitted Andean hat with ear flaps and a small pompom",
     "a knitted hat with ear flaps", False),
    (r"\bponchos?\b",
     "a poncho is a square woven cloak with a hole for the head and simple geometric stripes",
     "a poncho", False),
    (r"\b(maize|corn)\b|\bsara\b",
     "Andean maize plants are tall stalks with long leaves and big cobs",
     "maize plants or cobs", False),
    (r"\b(reed boats?|totora)\b",
     "a totora reed boat is made of bundled reeds tied together, with a curved raised bow",
     "a boat made of bundled reeds", False),
]

# Abstract things the planner sometimes lists as "elements" although they can't be checked
_ABSTRACT = re.compile(r"^(the )?((morning|night|evening|blue|clear|open)\s+)?(sky|background|landscape|"
                       r"ground|scenery|air|light|atmosphere|horizon|scene|nature|day|night|morning)$", re.I)


def is_andean(culture: str) -> bool:
    return (culture or "").strip().lower().startswith(("andean", "andino", "inca", "quechua", "aymara"))


def _matches(text: str, culture: str):
    t = (text or "").lower()
    for pat, desc, hint, andean_only in GLOSSARY:
        if andean_only and not is_andean(culture):
            continue
        if re.search(pat, t):
            yield pat, desc, hint


def enrich_scene(scene: str, elements: Optional[List[str]] = None, culture: str = "") -> str:
    """Append the precise look of every glossary subject that appears in the scene or elements."""
    scene = " ".join((scene or "").split()).rstrip(".")
    src = " ".join([scene] + list(elements or []))
    extra = [desc for _, desc, _ in _matches(src, culture)]
    if not extra:
        return scene
    return scene + ". " + ". ".join(d[0].upper() + d[1:] for d in extra)


def verify_hint(element: str, culture: str = "") -> str:
    """What the checker must see for this element ('' = the element name is enough)."""
    for _, _, hint in _matches(element, culture):
        return hint
    return ""


def drawable_elements(elements: List[str]) -> List[str]:
    """Drop abstract elements ("sky", "morning sky", "landscape") unless nothing else is left."""
    keep = [e for e in elements if not _ABSTRACT.match((e or "").strip())]
    return keep or list(elements)
