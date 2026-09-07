"""Story analysis: transcript -> title, elements, scene, image prompt, narration.

    analyze(text, lang=None, backends=None) -> dict

Backends (config.STORY_BACKENDS, tried in order until one works):
    remote     laptop/ai_server.py  POST /analyze
    ollama     Ollama directly       POST /api/generate (JSON mode)
    anthropic  Claude API            needs internet + ANTHROPIC_API_KEY
    rules      offline keyword rules, never fails
"""
import json
import logging
import re

import config
from language import detect_language, find_elements

log = logging.getLogger("story")

ENGLISH = {
    "person": "a child in a poncho", "condor": "a condor flying", "llama": "a llama", "puma": "a puma",
    "cat": "a cat", "dog": "a dog", "bird": "a bird", "fish": "a fish", "bear": "a spectacled bear",
    "fox": "a fox", "snake": "a snake", "mountain": "Andean mountains", "river": "a river",
    "lake": "a lake", "sea": "the sea", "sun": "the sun", "moon": "the moon", "star": "stars",
    "cloud": "clouds", "rain": "rain", "tree": "a tree", "flower": "flowers", "corn": "corn plants",
    "potato": "potatoes", "field": "farm fields", "house": "an adobe house", "bridge": "a rope bridge",
    "path": "a mountain path", "temple": "an Inca temple", "boat": "a reed boat", "fire": "a bonfire",
    "music": "a quena flute",
}
SPANISH = {
    "person": "una niña con poncho", "condor": "un cóndor", "llama": "una llama", "puma": "un puma",
    "cat": "un gato", "dog": "un perro", "bird": "un pájaro", "fish": "un pez", "bear": "un oso de anteojos",
    "fox": "un zorro", "snake": "una serpiente", "mountain": "las montañas", "river": "un río",
    "lake": "una laguna", "sea": "el mar", "sun": "el sol", "moon": "la luna", "star": "las estrellas",
    "cloud": "las nubes", "rain": "la lluvia", "tree": "un árbol", "flower": "flores", "corn": "maíz",
    "potato": "papas", "field": "la chacra", "house": "una casa de adobe", "bridge": "un puente",
    "path": "un camino", "temple": "un templo", "boat": "un bote de totora", "fire": "una fogata",
    "music": "una quena",
}

_JSON_INSTRUCTIONS = (
    "You analyse a short oral story told to a drawing robot in Peru. Reply with ONLY a JSON object with keys: "
    '"title" (short Spanish title), "elements" (list of keys from this set: %s), '
    '"scene" (one Spanish sentence describing the picture to draw), '
    '"image_prompt" (one English sentence describing a simple black-ink line drawing of the scene, no text), '
    '"narration" (2-3 warm Spanish sentences retelling the story).' % ", ".join(ENGLISH.keys())
)


def _title_from(text):
    words = re.findall(r"[\wñáéíóúü']+", text)
    t = " ".join(words[:6]).strip()
    return (t[0].upper() + t[1:]) if t else "Una historia"


def _join_es(names):
    if not names:
        return "un paisaje andino"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " y " + names[-1]


def _join_en(names):
    if not names:
        return "an Andean landscape"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _rules(text, lang):
    elements = find_elements(text) or ["mountain", "sun", "person"]
    es = [SPANISH.get(e, e) for e in elements]
    en = [ENGLISH.get(e, e) for e in elements]
    scene = f"Una escena andina con {_join_es(es)}."
    image_prompt = f"{_join_en(en)}, in the Andes"
    short = text.strip()
    if len(short) > 220:
        short = short[:217].rsplit(" ", 1)[0] + "..."
    if lang == "quechua":
        narration = f"Esta historia fue contada en quechua. Habla de {_join_es(es)}. {short}"
    else:
        narration = f"Esta es la historia de {_join_es(es)}. {short}"
    return {"title": _title_from(text), "elements": elements, "scene": scene,
            "image_prompt": image_prompt, "narration": narration}


def _validate(d, text, lang):
    base = _rules(text, lang)
    if not isinstance(d, dict):
        raise ValueError("not a JSON object")
    out = dict(base)
    for k in ("title", "scene", "image_prompt", "narration"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    els = d.get("elements")
    if isinstance(els, list):
        els = [e for e in els if isinstance(e, str) and e in ENGLISH]
        if els:
            out["elements"] = els
    return out


def _remote(text, lang):
    import requests
    r = requests.post(f"{config.AI_SERVER_URL.rstrip('/')}/analyze", json={"text": text, "lang": lang},
                      timeout=(3, config.AI_SERVER_TIMEOUT))
    r.raise_for_status()
    return _validate(r.json(), text, lang)


def _extract_json(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    return json.loads(m.group(0))


def _ollama(text, lang):
    import requests
    prompt = f"{_JSON_INSTRUCTIONS}\n\nStory ({lang}):\n{text}"
    r = requests.post(f"{config.OLLAMA_URL.rstrip('/')}/api/generate",
                      json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
                      timeout=(3, config.AI_SERVER_TIMEOUT))
    r.raise_for_status()
    return _validate(_extract_json(r.json().get("response", "")), text, lang)


def _anthropic(text, lang):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(model=config.ANTHROPIC_MODEL, max_tokens=600,
                                 system=_JSON_INSTRUCTIONS,
                                 messages=[{"role": "user", "content": f"Story ({lang}):\n{text}"}])
    return _validate(_extract_json("".join(b.text for b in msg.content if hasattr(b, "text"))), text, lang)


BACKENDS = {"remote": _remote, "ollama": _ollama, "anthropic": _anthropic, "rules": _rules}


def analyze(text, lang=None, backends=None, on_status=None):
    text = (text or "").strip()
    if lang is None:
        lang, _ = detect_language(text)
    errors = []
    for name in (backends or config.STORY_BACKENDS):
        fn = BACKENDS.get(name)
        if fn is None:
            continue
        try:
            if on_status:
                on_status(name)
            out = fn(text, lang)
            out.update({"backend": name, "lang": lang, "text": text, "errors": errors})
            return out
        except Exception as e:
            log.warning("story backend '%s' failed: %s", name, e)
            errors.append(f"{name}: {e}")
    out = _rules(text, lang)
    out.update({"backend": "rules", "lang": lang, "text": text, "errors": errors})
    return out
