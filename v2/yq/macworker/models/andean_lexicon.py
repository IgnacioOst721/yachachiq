"""Small Quechua / Aymara dictionary of story words, so machine translation can't lose the story.

NLLB-200 turned "Ñawpa pachapi huk atuqmi karqan... qucha patapi takirqan" (there was a fox... it sang
by the lake) into "había un grupo... cantaba sobre el mar" (2026-09-27): the fox vanished and the
drawing showed a crowd by the sea. The words a drawing depends on (animals, nature, people, places,
common verbs) are few and well known, so they are looked up here and given to the planner and to the
translation post-editor as reliable meanings.

    hits(text, lang)       -> [(word_in_text, root, es, en), ...]
    hint_lines(hits)       -> "atuq = zorro (fox)" lines for an LLM prompt

Quechua is agglutinative: a word matches a root when it STARTS with it (atuq-mi, killa-ta, taki-rqan).
Only roots of 3+ letters are used, to avoid matching inside unrelated words.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# Southern Quechua (Cusco-Collao / Ayacucho-Chanka spellings; both listed when they differ)
QUECHUA = [
    # animals
    ("atuq", "zorro", "fox"), ("atoq", "zorro", "fox"), ("kuntur", "cóndor", "Andean condor"),
    ("llama", "llama", "llama"), ("paqucha", "alpaca", "alpaca"), ("paqocha", "alpaca", "alpaca"),
    ("allpaka", "alpaca", "alpaca"), ("wik'uña", "vicuña", "vicuña"), ("wikuña", "vicuña", "vicuña"),
    ("puma", "puma", "puma"), ("uturunku", "jaguar", "jaguar"), ("otorongo", "jaguar", "jaguar"),
    ("ukumari", "oso andino", "spectacled bear"), ("ukuku", "oso andino", "spectacled bear"),
    ("allqu", "perro", "dog"), ("allqo", "perro", "dog"), ("misi", "gato", "cat"), ("michi", "gato", "cat"),
    ("wallpa", "gallina", "hen"), ("quwi", "cuy", "guinea pig"), ("qowi", "cuy", "guinea pig"),
    ("pisqu", "pájaro", "bird"), ("pisqo", "pájaro", "bird"), ("pishqu", "pájaro", "bird"),
    ("challwa", "pez", "fish"), ("amaru", "serpiente (amaru)", "serpent"), ("machaqway", "culebra", "snake"),
    ("hamp'atu", "sapo", "toad"), ("taruka", "venado", "deer"), ("wisk'acha", "vizcacha", "vizcacha"),
    ("wiskacha", "vizcacha", "vizcacha"), ("q'inti", "picaflor", "hummingbird"), ("añas", "zorrillo", "skunk"),
    ("anka", "águila", "eagle"), ("waka", "vaca", "cow"), ("uwija", "oveja", "sheep"), ("kawallu", "caballo", "horse"),
    # nature and sky
    ("killa", "luna", "moon"), ("inti", "sol", "sun"), ("quyllur", "estrella", "star"),
    ("qoyllur", "estrella", "star"), ("ch'aska", "lucero", "morning star"), ("qucha", "laguna, lago", "lake, lagoon"),
    ("qocha", "laguna, lago", "lake, lagoon"), ("mayu", "río", "river"), ("urqu", "cerro, montaña", "mountain, hill"),
    ("orqo", "cerro, montaña", "mountain, hill"), ("apu", "apu (montaña sagrada)", "apu (sacred mountain)"),
    ("rit'i", "nieve", "snow"), ("riti", "nieve", "snow"), ("para", "lluvia", "rain"), ("wayra", "viento", "wind"),
    ("phuyu", "nube", "cloud"), ("sach'a", "árbol", "tree"), ("sacha", "árbol", "tree"), ("t'ika", "flor", "flower"),
    ("tika", "flor", "flower"), ("rumi", "piedra", "stone"), ("yaku", "agua", "water"), ("unu", "agua", "water"),
    ("nina", "fuego", "fire"), ("illapa", "rayo", "lightning"), ("k'uychi", "arcoíris", "rainbow"),
    ("pachamama", "Pachamama (Madre Tierra)", "Pachamama (Mother Earth)"), ("tuta", "noche", "night"),
    ("p'unchaw", "día", "day"), ("punchaw", "día", "day"), ("pampa", "pampa (llanura)", "plain"),
    ("sara", "maíz", "maize"), ("papa", "papa", "potato"), ("kinwa", "quinua", "quinoa"), ("kuka", "coca", "coca leaf"),
    # people
    ("warmi", "mujer", "woman"), ("qhari", "hombre", "man"), ("wawa", "bebé, niño", "baby, small child"), ("warma", "niño, niña", "child"),
    ("pasña", "niña", "girl"), ("wayna", "joven, muchacho", "young man"), ("sipas", "muchacha", "young woman"),
    ("machula", "abuelo", "grandfather"), ("machu", "anciano", "old man"), ("paya", "anciana", "old woman"),
    ("awicha", "abuela", "grandmother"), ("mama", "madre, mamá", "mother, mom"), ("tayta", "padre, papá", "father, dad"),
    ("runa", "persona, gente", "person, people"), ("ayllu", "familia, comunidad", "family, community"),
    # places and things
    ("wasi", "casa", "house"), ("llaqta", "pueblo, ciudad", "town, village"), ("chakra", "chacra, campo", "farm field"), ("ñan", "camino", "path"),
    ("chaka", "puente", "bridge"), ("pata", "orilla, borde", "shore, edge"),
    # verbs (root + typical suffixes)
    ("taki", "cantar", "to sing"), ("tusu", "bailar", "to dance"), ("phaw", "volar", "to fly"), ("puri", "caminar", "to walk"),
    ("mikhu", "comer", "to eat"), ("puñu", "dormir", "to sleep"), ("waqa", "llorar", "to cry"), ("asi", "reír", "to laugh"),
    ("qhawa", "mirar", "to look"), ("riku", "ver", "to see"), ("tiya", "vivir, estar sentado", "to live, to sit"),
]

AYMARA = [
    ("qamaqi", "zorro", "fox"), ("kunturi", "cóndor", "Andean condor"), ("qarwa", "llama", "llama"),
    ("allpachu", "alpaca", "alpaca"), ("wari", "vicuña", "vicuña"), ("anu", "perro", "dog"), ("phisi", "gato", "cat"),
    ("jamach'i", "pájaro", "bird"), ("challwa", "pez", "fish"), ("phaxsi", "luna", "moon"), ("inti", "sol", "sun"),
    ("quta", "lago, laguna", "lake"), ("jawira", "río", "river"), ("qullu", "cerro, montaña", "mountain, hill"), ("jallu", "lluvia", "rain"),
    ("panqara", "flor", "flower"), ("nina", "fuego", "fire"), ("uma", "agua", "water"), ("uta", "casa", "house"),
    ("marka", "pueblo", "town"), ("warmi", "mujer", "woman"), ("chacha", "hombre", "man"), ("wawa", "bebé", "baby"),
    ("jaqi", "persona", "person"), ("ch'uqi", "papa", "potato"), ("tunqu", "maíz", "maize"), ("jupha", "quinua", "quinoa"),
    ("kuka", "coca", "coca leaf"),
]

_WORD = re.compile(r"[a-zñáéíóúü'’]+", re.I)


def _norm(s: str) -> str:
    s = (s or "").lower().replace("’", "'")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn" or c == "̃")


def _table(lang: str):
    c = (lang or "").split("_")[0]
    if c in ("ayr", "ayc", "aym"):
        return AYMARA
    if c.startswith(("qu", "qv", "qw", "qx")):
        return QUECHUA
    return []


def hits(text: str, lang: str) -> List[Tuple[str, str, str, str]]:
    """Dictionary words found in text (first occurrence of each root, longest root first)."""
    table = sorted(_table(lang), key=lambda r: -len(r[0]))
    found, seen = [], set()
    for w in _WORD.findall(text or ""):
        nw = _norm(w)
        for root, es, en in table:
            nr = _norm(root)
            if len(nr) >= 3 and nw.startswith(nr):
                if (es, en) not in seen:
                    seen.add((es, en))
                    found.append((w, root, es, en))
                break
    return found


def hint_lines(found) -> str:
    return "\n".join("%s = %s (%s)" % (w, es, en) for w, _r, es, en in found)


POSTEDIT_SYSTEM = {
    "spa": ("Eres un traductor cuidadoso. Te doy un texto original en {src}, un borrador de traducción automática "
            "al castellano y un glosario CONFIABLE con el significado de algunas palabras del original. Corrige el "
            "borrador para que cada palabra del glosario aparezca con ese significado: si el glosario dice "
            "'atuqmi = zorro', la traducción TIENE que hablar de un zorro, y lo que el borrador puso en su lugar (por "
            "ejemplo 'un grupo') se cambia. Cambia también lo que no está en el original. Deja igual todo lo demás. "
            "No agregues ideas ni adornos. Responde solo con la traducción corregida."),
    "eng": ("You are a careful translator. You get an original text in {src}, a machine-translation draft into "
            "English and a RELIABLE glossary of some words of the original. Every glossary word MUST appear with that "
            "meaning: if the glossary says 'atuqmi = fox', the translation MUST talk about a fox, replacing whatever "
            "the draft put instead (e.g. 'a group'). Also remove what is not in the original. Keep everything else. "
            "Add nothing. Reply with the corrected translation only."),
}
_LANG_NAME = {"que": "quechua", "aym": "aimara"}


def llm_postedit(src_text: str, src: str, draft: str, tgt: str, found=None) -> str:
    """Fix a machine translation with the dictionary using the LLM; returns the draft unchanged when the
    LLM is not available here or its answer looks wrong. Only Quechua/Aymara -> Spanish/English."""
    tf = (tgt or "").split("_")[0]
    table = _table(src)
    if tf not in POSTEDIT_SYSTEM or not table:
        return draft
    found = hits(src_text, src) if found is None else found
    if not found:
        return draft
    from yq.common import config
    if config.mock("llm"):
        return draft
    try:
        from . import llm
    except Exception:
        return draft
    fam = "aym" if table is AYMARA else "que"
    gloss = "\n".join("%s = %s" % (w, es if tf == "spa" else en) for w, _r, es, en in found)
    msgs = [{"role": "system", "content": POSTEDIT_SYSTEM[tf].format(src=_LANG_NAME[fam])},
            {"role": "user", "content": "Original:\n%s\n\nBorrador / draft:\n%s\n\nGlosario / glossary:\n%s"
                                        % (src_text.strip(), draft.strip(), gloss)}]
    # nouns (animals, nature, people, places) must survive; verbs change form, so they are not checked
    # any of the listed synonyms counts ("cerro, montaña": either word is fine)
    must = [[a.strip().lower() for a in (es if tf == "spa" else en).split(" (")[0].split(",") if a.strip()]
            for _w, r, es, en in found if not _is_verb(r)]
    best = draft
    for attempt in range(2):
        try:
            out = llm.chat(msgs, max_tokens=max(120, int(len(draft) / 2)), temperature=0.0)
        except Exception:
            return best
        out = " ".join((out or "").strip().strip('"«»“”').split())
        if not out or not (0.5 <= len(out) / max(1, len(draft)) <= 2.0):
            continue
        missing = [alts[0] for alts in must if not any(_stem(a) in _norm(out) for a in alts)]
        best = out
        if not missing:
            return out
        msgs = msgs + [{"role": "assistant", "content": out},
                       {"role": "user", "content": ("Falta: %s. Corrígelo." if tf == "spa" else "Missing: %s. Fix it.")
                                                   % ", ".join(missing)}]
    return best


_VERB_ROOTS = {"taki", "tusu", "phaw", "puri", "mikhu", "puñu", "waqa", "asi", "qhawa", "riku", "tiya"}


def _is_verb(root: str) -> bool:
    return root in _VERB_ROOTS


def _stem(word: str) -> str:
    """'zorro' -> 'zorr', 'laguna' -> 'lagun': matches zorro/zorros/zorrito, laguna/lagunas."""
    w = _norm(word)
    return w[:-1] if len(w) > 4 else w
