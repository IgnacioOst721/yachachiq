"""Language detection (Spanish / Quechua) and story words -> motif keys.

detect_language(text) -> ("quechua" | "spanish", matched_words)
find_elements(text)   -> ordered list of motif keys found in the story
"""
import re
import unicodedata

# Common Quechua words. Two or more hits = the story is (at least partly) Quechua.
QUECHUA_WORDS = {
    "ñuqa", "nuqa", "qam", "pay", "ñuqanchik", "rimani", "rimay", "runa", "simi",
    "tayta", "mama", "wawa", "inti", "killa", "pacha", "yaku", "mayu", "urqu",
    "wasi", "allqu", "misi", "kuntur", "puma", "llama", "sara", "papa", "sunqu",
    "allin", "sumaq", "hatun", "huchuy", "kawsay", "kay", "chay", "ima", "pi",
    "wayra", "nina", "quyllur", "ch'aska", "chaska", "apu", "ayllu", "chakra",
    "wiraqucha", "willka", "qucha", "sacha", "t'ika", "tika", "challwa", "wallpa",
    "ukuku", "atuq", "wik'uña", "wikuña", "paqucha", "pisqu", "hamp'atu",
}

# word (stem) -> motif key.  Stems are matched as prefixes on accent-stripped
# lowercase words, so "montañas", "montaña", "montanas" all hit "mountain".
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

# Keys that "nin" must not match (spanish 'ninguno' etc.) are rare enough; but
# never let "mar" match "martes"/"maria": handled by the exact-list below.
EXACT_ONLY = {"mar", "sol", "ave", "oso", "pez", "apu", "man", "boy", "cat", "dog"}


def _norm(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c) or c in "̃")
    # keep ñ: NFKD splits it into n + combining tilde; put it back
    text = unicodedata.normalize("NFC", text)
    return text


def _words(text):
    return re.findall(r"[a-zñ']+", _norm(text))


def detect_language(text):
    words = _words(text)
    hits = [w for w in words if w in QUECHUA_WORDS]
    if len(hits) >= 2:
        return "quechua", hits
    return "spanish", []


def find_elements(text):
    found = []
    for w in _words(text):
        for stem, key in ELEMENT_STEMS:
            if stem in EXACT_ONLY:
                ok = (w == stem) or (w == stem + "es") or (w == stem + "s")
            else:
                ok = w.startswith(stem)
            if ok and key not in found:
                found.append(key)
                break
    return found


if __name__ == "__main__":
    import sys
    t = " ".join(sys.argv[1:]) or "Un cóndor volaba sobre las montañas mientras una niña y su llama iban al río bajo el sol"
    print(detect_language(t))
    print(find_elements(t))
