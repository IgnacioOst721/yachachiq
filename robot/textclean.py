"""Tidy up a raw story before it is analysed, drawn and published.

Signed stories arrive as bare letters with no case, accents or punctuation
("MI ABUELA VIVIA EN LA SIERRA"); dictated ones often miss a final full stop.
clean(text) -> (clean_text, notes)

Two layers:
  rules  always available, offline, conservative: sentence case, spacing,
         punctuation, and accents only for words where Spanish leaves no doubt
  llm    when the laptop/Ollama/Claude backend answers, it rewrites the text
         properly (accents, commas) and its result is used if it is clearly the
         same story - never a different one
"""
import logging
import re
import unicodedata

import config

log = logging.getLogger("textclean")

# Words whose accented form is the only real option in Spanish. Ambiguous pairs
# (papa/papá, esta/está, el/él, si/sí, mas/más, como/cómo...) are NOT here: a
# wrong accent changes the meaning, so those are left to the LLM.
ACENTOS = {
    "vivia": "vivía", "vivian": "vivían", "habia": "había", "habian": "habían",
    "decia": "decía", "decian": "decían", "queria": "quería", "querian": "querían",
    "tenia": "tenía", "tenian": "tenían", "sabia": "sabía", "sabian": "sabían",
    "veia": "veía", "veian": "veían", "solia": "solía", "solian": "solían",
    "iba": "iba", "corria": "corría", "corrian": "corrían", "subia": "subía",
    "bajaba": "bajaba", "volvia": "volvía", "sentia": "sentía", "traia": "traía",
    "montaña": "montaña", "montañas": "montañas", "niña": "niña", "niño": "niño",
    "niñas": "niñas", "niños": "niños", "años": "años", "año": "año",
    "rio": "río", "dia": "día", "dias": "días", "despues": "después",
    "tambien": "también", "ademas": "además", "asi": "así", "aqui": "aquí",
    "alli": "allí", "ahi": "ahí", "detras": "detrás", "jamas": "jamás",
    "quiza": "quizá", "corazon": "corazón", "cancion": "canción",
    "condor": "cóndor", "condores": "cóndores", "leon": "león",
    "arbol": "árbol", "arboles": "árboles", "pajaro": "pájaro", "pajaros": "pájaros",
    "musica": "música", "ultimo": "último", "ultima": "última", "unico": "único",
    "facil": "fácil", "dificil": "difícil", "joven": "joven", "jovenes": "jóvenes",
    "mama": "mamá", "abuelita": "abuelita", "peru": "Perú", "peruano": "peruano",
    "america": "América", "andes": "Andes", "cusco": "Cusco", "lima": "Lima",
    "quechua": "quechua", "titicaca": "Titicaca", "machu": "Machu",
}

FIN_DE_FRASE = re.compile(r"[.!?…]$")


def _es_mayusculas(t):
    letras = [c for c in t if c.isalpha()]
    return bool(letras) and sum(1 for c in letras if c.isupper()) / len(letras) > 0.85


def _acentuar(palabra):
    base = palabra.lower()
    fijo = ACENTOS.get(base)
    if not fijo:
        return palabra
    if palabra.isupper() and len(palabra) > 1:
        return fijo.upper()
    if palabra[:1].isupper():
        return fijo[:1].upper() + fijo[1:]
    return fijo


def rules(text):
    """Offline clean-up. Never invents or drops words."""
    t = " ".join((text or "").split())
    if not t:
        return t, []
    notas = []
    if _es_mayusculas(t):
        t = t.lower()
        notas.append("de MAYÚSCULAS a minúsculas")
    partes = re.split(r"(\W+)", t)
    cambio_acento = 0
    for i, p in enumerate(partes):
        if p.isalpha():
            nuevo = _acentuar(p)
            if nuevo != p:
                partes[i] = nuevo
                cambio_acento += 1
    t = "".join(partes)
    if cambio_acento:
        notas.append(f"{cambio_acento} tilde(s)")
    # espacios antes de la puntuación y después de ella
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,.;:!?])(?=[^\s\d])", r"\1 ", t)
    # mayúscula al inicio y después de punto
    def _mayus(m):
        return m.group(1) + m.group(2).upper()
    t = re.sub(r"(^|[.!?…]\s+)([a-záéíóúüñ])", _mayus, t)
    if not FIN_DE_FRASE.search(t):
        t += "."
        notas.append("punto final")
    return t, notas


def _parecida(a, b):
    """Same story? compares the bag of words ignoring case and accents."""
    def norm(s):
        s = unicodedata.normalize("NFKD", s.lower())
        s = "".join(c for c in s if not unicodedata.combining(c))
        return set(re.findall(r"[a-z0-9ñ]+", s))
    wa, wb = norm(a), norm(b)
    if not wa:
        return False
    return len(wa & wb) / len(wa) >= 0.7 and 0.5 <= len(wb) / max(len(wa), 1) <= 1.8


def llm(text, lang="spanish"):
    """Ask the laptop/Ollama/Claude to fix spelling and punctuation. Returns None if it cannot."""
    prompt = ("Corrige ortografía, tildes y puntuación del siguiente texto en "
              + ("quechua/español" if lang == "quechua" else "español") +
              ". NO cambies las palabras, no agregues ni quites ideas, no lo traduzcas, "
              "no lo resumas. Responde SOLO con el texto corregido.\n\n" + text)
    for backend in config.STORY_BACKENDS:
        try:
            if backend == "remote":
                import requests
                r = requests.post(f"{config.AI_SERVER_URL.rstrip('/')}/correct",
                                  json={"text": text, "lang": lang}, timeout=(3, 45))
                r.raise_for_status()
                out = (r.json() or {}).get("text", "")
            elif backend == "ollama":
                import requests
                r = requests.post(f"{config.OLLAMA_URL.rstrip('/')}/api/generate",
                                  json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
                                  timeout=(3, 60))
                r.raise_for_status()
                out = r.json().get("response", "")
            elif backend == "anthropic":
                import anthropic
                m = anthropic.Anthropic().messages.create(
                    model=config.ANTHROPIC_MODEL, max_tokens=700,
                    messages=[{"role": "user", "content": prompt}])
                out = "".join(b.text for b in m.content if hasattr(b, "text"))
            else:
                continue
            out = " ".join((out or "").split()).strip()
            if out and _parecida(text, out):
                return out
            if out:
                log.warning("corrector: respuesta descartada (no es la misma historia)")
        except Exception as e:
            log.info("corrector '%s' no disponible: %s", backend, e)
    return None


def clean(text, lang="spanish"):
    """Returns (texto_limpio, notas). Never raises; worst case gives the text back."""
    try:
        base, notas = rules(text)
        if not config.CLEAN_WITH_LLM or not base:
            return base, notas
        mejor = llm(base, lang)
        if mejor:
            fin, _ = rules(mejor)
            return fin, notas + ["corregido con IA"]
        return base, notas
    except Exception as e:
        log.warning("clean failed: %s", e)
        return text, []
