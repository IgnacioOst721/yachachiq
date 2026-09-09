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


# Concrete words that actually appear in the stories people tell the robot, so the offline
# analysis can draw what was said instead of a generic Andean postcard. Keys are accent-free
# and matched on word stems, values are what goes into the English drawing prompt.
LEXICON = {
    # familia y gente
    "abuelo": "an old man", "abuela": "an old woman", "abuelita": "an old woman",
    "madre": "a mother", "mama": "a mother", "padre": "a father", "papa_": "a father",
    "nino": "a child", "nina": "a girl", "hijo": "a child", "hija": "a girl",
    "hermano": "a brother", "hermana": "a sister", "nieto": "a grandchild", "nieta": "a grandchild",
    "familia": "a family", "amigo": "a friend", "hombre": "a man", "mujer": "a woman",
    "pastor": "a shepherd", "viajero": "a traveller", "tejedora": "a weaver",
    # animales
    "llama": "a llama", "alpaca": "an alpaca", "vicuna": "a vicuna", "oveja": "sheep",
    "condor": "a condor", "zorro": "a fox", "puma": "a puma", "oso": "a spectacled bear",
    "perro": "a dog", "gato": "a cat", "pajaro": "a bird", "ave": "a bird", "pez": "a fish",
    "serpiente": "a snake", "culebra": "a snake", "vaca": "a cow", "toro": "a bull",
    "caballo": "a horse", "burro": "a donkey", "gallina": "a hen", "rana": "a frog",
    "mariposa": "a butterfly", "colibri": "a hummingbird", "aguila": "an eagle", "raton": "a mouse",
    "jaguar": "a jaguar", "otorongo": "a jaguar", "mono": "a monkey", "tortuga": "a turtle",
    "cuy": "a guinea pig", "chancho": "a pig", "pato": "a duck", "abeja": "bees", "arana": "a spider",
    # paisaje
    "montana": "Andean mountains", "cerro": "a hill", "nevado": "a snowy peak", "cordillera": "a mountain range",
    "rio": "a river", "laguna": "a lagoon", "lago": "a lake", "mar": "the sea", "playa": "a beach",
    "valle": "a valley", "selva": "the jungle", "bosque": "a forest", "desierto": "a desert",
    "arbol": "a tree", "flor": "flowers", "piedra": "stones", "cueva": "a cave",
    "camino": "a path", "puente": "a rope bridge", "chacra": "farm fields", "campo": "fields",
    "pueblo": "a village", "ciudad": "a town", "sierra": "the highlands", "pampa": "an open plain",
    # cielo y clima
    "sol": "the sun", "luna": "the moon", "estrella": "stars", "nube": "clouds",
    "lluvia": "rain", "viento": "wind", "nieve": "snow", "tormenta": "a storm",
    "arcoiris": "a rainbow", "noche": "night", "amanecer": "sunrise", "atardecer": "sunset",
    # cosas
    "casa": "an adobe house", "choza": "a hut", "puerta": "a door", "ventana": "a window",
    "olla": "a clay pot", "pan": "bread", "manta": "a woven blanket", "poncho": "a poncho",
    "sombrero": "a hat", "canasta": "a basket", "telar": "a loom", "quena": "a flute",
    "tambor": "a drum", "charango": "a charango", "bote": "a reed boat", "barco": "a boat",
    "fuego": "a fire", "fogata": "a bonfire", "vela": "a candle", "libro": "a book",
    "maiz": "corn plants", "quinua": "quinoa", "trigo": "wheat", "coca": "coca leaves",
    "templo": "an Inca temple", "iglesia": "a church", "escuela": "a school", "mercado": "a market",
    # acciones (dan movimiento al dibujo)
    "tejer": "weaving", "tejia": "weaving", "camina": "walking", "corr": "running",
    "vola": "flying", "nada": "swimming", "duerme": "sleeping", "dormia": "sleeping",
    "canta": "singing", "baila": "dancing", "siembra": "planting", "cosecha": "harvesting",
    "pasta": "grazing", "cocina": "cooking", "llora": "crying", "rie": "laughing",
    "sube": "climbing", "baja": "going down", "pesca": "fishing", "monta": "riding",
}
_ACENTOS = str.maketrans("áéíóúü", "aeiouu")


def _content(text, limit=7):
    """Words from the story itself, translated for the drawing prompt, in the order they appear."""
    plain = text.lower().translate(_ACENTOS)
    words = re.findall(r"[a-zñ]+", plain)
    out, seen = [], set()
    for w in words:
        for key, en in LEXICON.items():
            k = key.rstrip("_")
            if (w == k or w == k + "s" or w == k + "es" or (len(k) > 4 and w.startswith(k))) and en not in seen:
                seen.add(en)
                out.append(en)
                break
        if len(out) >= limit:
            break
    return out


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
    """Offline analysis. Builds the picture from what the story actually says; only falls back
    to a generic Andean scene when no known word appears at all."""
    found = find_elements(text)
    words = _content(text)
    elements = found or ["mountain", "sun", "person"]
    es = [SPANISH.get(e, e) for e in found]

    if words:
        image_prompt = ", ".join(words) + ", in the Andes"
        scene = "Una escena de la historia: " + _join_es(es) + "." if es else "Una escena de la historia."
    else:
        image_prompt = f"{_join_en([ENGLISH.get(e, e) for e in elements])}, in the Andes"
        scene = f"Una escena andina con {_join_es([SPANISH.get(e, e) for e in elements])}."

    short = text.strip()
    if len(short) > 220:
        short = short[:217].rsplit(" ", 1)[0] + "..."
    # Without a language model the honest retelling is the story itself: adding "una historia de
    # X" invented characters (a story about a grandmother came back as "una niña con poncho").
    narration = f"Contada en quechua. {short}" if lang == "quechua" else short
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
