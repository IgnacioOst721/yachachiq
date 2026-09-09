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
    # cada palabra -> (para el dibujo en ingles, para mostrar en pantalla en espanol)
    # familia y gente
    "abuelo": ("an old man", "un abuelo"), "abuela": ("an old woman", "una abuela"),
    "abuelita": ("an old woman", "una abuelita"), "madre": ("a mother", "una madre"),
    "mama": ("a mother", "una mamá"), "padre": ("a father", "un padre"),
    "nino": ("a child", "un niño"), "nina": ("a girl", "una niña"),
    "hijo": ("a child", "un hijo"), "hija": ("a girl", "una hija"),
    "hermano": ("a brother", "un hermano"), "hermana": ("a sister", "una hermana"),
    "nieto": ("a grandchild", "un nieto"), "nieta": ("a grandchild", "una nieta"),
    "familia": ("a family", "una familia"), "amigo": ("a friend", "un amigo"),
    "hombre": ("a man", "un hombre"), "mujer": ("a woman", "una mujer"),
    "pastor": ("a shepherd", "un pastor"), "viajero": ("a traveller", "un viajero"),
    "tejedora": ("a weaver", "una tejedora"),
    # animales
    "llama": ("a llama", "una llama"), "alpaca": ("an alpaca", "una alpaca"),
    "vicuna": ("a vicuna", "una vicuña"), "oveja": ("sheep", "ovejas"),
    "condor": ("a condor", "un cóndor"), "zorro": ("a fox", "un zorro"),
    "puma": ("a puma", "un puma"), "oso": ("a spectacled bear", "un oso"),
    "perro": ("a dog", "un perro"), "gato": ("a cat", "un gato"),
    "pajaro": ("a bird", "un pájaro"), "ave": ("a bird", "un ave"),
    "pez": ("a fish", "un pez"), "serpiente": ("a snake", "una serpiente"),
    "culebra": ("a snake", "una culebra"), "vaca": ("a cow", "una vaca"),
    "toro": ("a bull", "un toro"), "caballo": ("a horse", "un caballo"),
    "burro": ("a donkey", "un burro"), "gallina": ("a hen", "una gallina"),
    "rana": ("a frog", "una rana"), "mariposa": ("a butterfly", "una mariposa"),
    "colibri": ("a hummingbird", "un colibrí"), "aguila": ("an eagle", "un águila"),
    "raton": ("a mouse", "un ratón"), "jaguar": ("a jaguar", "un jaguar"),
    "otorongo": ("a jaguar", "un otorongo"), "mono": ("a monkey", "un mono"),
    "tortuga": ("a turtle", "una tortuga"), "cuy": ("a guinea pig", "un cuy"),
    "chancho": ("a pig", "un chancho"), "pato": ("a duck", "un pato"),
    "abeja": ("bees", "abejas"), "arana": ("a spider", "una araña"),
    # paisaje
    "montana": ("Andean mountains", "las montañas"), "cerro": ("a hill", "un cerro"),
    "nevado": ("a snowy peak", "un nevado"), "cordillera": ("a mountain range", "la cordillera"),
    "rio": ("a river", "un río"), "laguna": ("a lagoon", "una laguna"),
    "lago": ("a lake", "un lago"), "mar": ("the sea", "el mar"),
    "playa": ("a beach", "una playa"), "valle": ("a valley", "un valle"),
    "selva": ("the jungle", "la selva"), "bosque": ("a forest", "un bosque"),
    "desierto": ("a desert", "el desierto"), "arbol": ("a tree", "un árbol"),
    "flor": ("flowers", "flores"), "piedra": ("stones", "piedras"),
    "cueva": ("a cave", "una cueva"), "camino": ("a path", "un camino"),
    "puente": ("a rope bridge", "un puente"), "chacra": ("farm fields", "la chacra"),
    "campo": ("fields", "el campo"), "pueblo": ("a village", "un pueblo"),
    "ciudad": ("a town", "una ciudad"), "sierra": ("the highlands", "la sierra"),
    "pampa": ("an open plain", "la pampa"),
    # cielo y clima
    "sol": ("the sun", "el sol"), "luna": ("the moon", "la luna"),
    "estrella": ("stars", "estrellas"), "nube": ("clouds", "nubes"),
    "lluvia": ("rain", "la lluvia"), "viento": ("wind", "el viento"),
    "nieve": ("snow", "la nieve"), "tormenta": ("a storm", "una tormenta"),
    "arcoiris": ("a rainbow", "un arcoíris"), "noche": ("night", "la noche"),
    "amanecer": ("sunrise", "el amanecer"), "atardecer": ("sunset", "el atardecer"),
    # cosas
    "casa": ("an adobe house", "una casa"), "choza": ("a hut", "una choza"),
    "puerta": ("a door", "una puerta"), "ventana": ("a window", "una ventana"),
    "olla": ("a clay pot", "una olla"), "pan": ("bread", "pan"),
    "manta": ("a woven blanket", "una manta"), "poncho": ("a poncho", "un poncho"),
    "sombrero": ("a hat", "un sombrero"), "canasta": ("a basket", "una canasta"),
    "telar": ("a loom", "un telar"), "quena": ("a flute", "una quena"),
    "tambor": ("a drum", "un tambor"), "charango": ("a charango", "un charango"),
    "bote": ("a reed boat", "un bote"), "barco": ("a boat", "un barco"),
    "fuego": ("a fire", "fuego"), "fogata": ("a bonfire", "una fogata"),
    "vela": ("a candle", "una vela"), "libro": ("a book", "un libro"),
    "maiz": ("corn plants", "maíz"), "quinua": ("quinoa", "quinua"),
    "trigo": ("wheat", "trigo"), "coca": ("coca leaves", "hojas de coca"),
    "templo": ("an Inca temple", "un templo"), "iglesia": ("a church", "una iglesia"),
    "escuela": ("a school", "una escuela"), "mercado": ("a market", "un mercado"),
    # acciones
    "tejer": ("weaving", "tejiendo"), "tejia": ("weaving", "tejiendo"),
    "camina": ("walking", "caminando"), "corr": ("running", "corriendo"),
    "vola": ("flying", "volando"), "nada": ("swimming", "nadando"),
    "duerme": ("sleeping", "durmiendo"), "dormia": ("sleeping", "durmiendo"),
    "canta": ("singing", "cantando"), "baila": ("dancing", "bailando"),
    "siembra": ("planting", "sembrando"), "cosecha": ("harvesting", "cosechando"),
    "pasta": ("grazing", "pastando"), "cocina": ("cooking", "cocinando"),
    "llora": ("crying", "llorando"), "rie": ("laughing", "riendo"),
    "sube": ("climbing", "subiendo"), "baja": ("going down", "bajando"),
    "pesca": ("fishing", "pescando"), "monta": ("riding", "montando"),
}
_ACENTOS = str.maketrans("áéíóúü", "aeiouu")


def _content(text, limit=7):
    """Words from the story itself, in the order they appear.
    Returns (para_el_dibujo_en_ingles, para_mostrar_en_espanol)."""
    plain = text.lower().translate(_ACENTOS)
    words = re.findall(r"[a-zn]+", plain)
    en, es, seen = [], [], set()
    for w in words:
        for key, (ingles, espanol) in LEXICON.items():
            k = key.rstrip("_")
            if (w == k or w == k + "s" or w == k + "es" or (len(k) > 4 and w.startswith(k))) and ingles not in seen:
                seen.add(ingles)
                en.append(ingles)
                es.append(espanol)
                break
        if len(en) >= limit:
            break
    return en, es


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
    words, palabras = _content(text)
    elements = found or ["mountain", "sun", "person"]
    es = [SPANISH.get(e, e) for e in found]

    if words:
        image_prompt = ", ".join(words) + ", in the Andes"
        # what the screen shows: the same things that went into the drawing, in Spanish
        scene = _join_es(palabras).capitalize() + "."
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
