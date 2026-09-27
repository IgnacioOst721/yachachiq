"""The visitor's "where was it found?" (CONTRACTS.md §9) as a bounded clue for identification.

parse_context() turns {"found_where", "region_hint", "notes", "lang"} into hints
{"country", "zone", "region", "site", "confidence", "source"} using the kiosk chip, a Peruvian
gazetteer (departments, valleys, sites) and ART's LLM when it is running. apply_prior() then
re-weights the retrieval culture scores with the EXACT rule documented in docs/box_analysis.md:

    m_c = B ** (w * k_c)       B = settings.CONTEXT_MAX_BOOST (1.5), w = hint confidence (0..1)
    k_c = +1 culture typical of the reported zone, +0.5 neighbouring zone / same country,
           0 unknown, -0.5 culture from another macro-region
    cultures whose image-only score is below 25 % of the best one can never be boosted (m_c <= 1),
    so the place can reorder close candidates but cannot create a culture the photos do not support.
"""
from __future__ import annotations

import re
from typing import Optional

from .taxonomy import fold, match_region

ZONES = ["costa_norte", "costa_central", "costa_sur", "sierra_norte", "sierra_central", "sierra_sur", "altiplano",
         "selva", "lima"]
ZONES_ES = {"costa_norte": "costa norte", "costa_central": "costa central", "costa_sur": "costa sur",
            "sierra_norte": "sierra norte", "sierra_central": "sierra central", "sierra_sur": "sierra sur",
            "altiplano": "altiplano", "selva": "selva", "lima": "Lima"}
NEIGHBOURS = {"costa_norte": ["sierra_norte", "costa_central"], "costa_central": ["lima", "costa_norte", "costa_sur", "sierra_central"],
              "lima": ["costa_central", "sierra_central"], "costa_sur": ["costa_central", "sierra_sur"],
              "sierra_norte": ["costa_norte", "sierra_central", "selva"], "sierra_central": ["sierra_norte", "sierra_sur", "costa_central", "lima"],
              "sierra_sur": ["sierra_central", "altiplano", "costa_sur"], "altiplano": ["sierra_sur"], "selva": ["sierra_norte"]}

# Curated Andean table: culture -> zones where its objects are typically found (core first).
CULTURE_ZONES = {
    "Cupisnique": ["costa_norte"], "Chavin": ["sierra_norte", "sierra_central", "costa_norte"], "Salinar": ["costa_norte"],
    "Gallinazo": ["costa_norte"], "Vicus": ["costa_norte"], "Moche": ["costa_norte"], "Lambayeque": ["costa_norte"],
    "Chimu": ["costa_norte", "costa_central"], "Recuay": ["sierra_norte"], "Paracas": ["costa_sur"], "Nasca": ["costa_sur"],
    "Ica": ["costa_sur"], "Chiribaya": ["costa_sur"], "Lima": ["lima", "costa_central"], "Chancay": ["lima", "costa_central"],
    "Wari": ["sierra_central", "sierra_sur", "costa_sur", "costa_central"], "Tiwanaku": ["altiplano"], "Pukara": ["altiplano"],
    "Aymara": ["altiplano"], "Inca": ["sierra_sur", "sierra_central", "altiplano", "costa_central", "costa_sur", "sierra_norte"],
    "Andean Colonial": ZONES, "Amazonian": ["selva"], "Diaguita": [], "Aguada": [], "Mapuche": [],
}
PERU_CULTURES = set(CULTURE_ZONES) - {"Diaguita", "Aguada", "Mapuche"}

# Gazetteer (folded names). Departments, provinces, valleys, cities and archaeological sites.
_GAZ = {
    "costa_norte": ["tumbes", "piura", "sullana", "vicus", "chulucanas", "lambayeque", "chiclayo", "sipan", "tucume", "batan grande",
                    "sican", "ferrenafe", "la libertad", "trujillo", "chan chan", "huaca de la luna", "huaca del sol", "huacas de moche",
                    "el brujo", "chicama", "jequetepeque", "san jose de moro", "pacasmayo", "viru", "santa", "chimbote", "casma",
                    "sechin", "nepena", "huanchaco", "pampa grande", "mochica"],
    "costa_central": ["huaura", "huacho", "caral", "supe", "barranca", "chancay", "huaral", "ancon", "canete", "mala", "lurin",
                      "pachacamac", "rimac", "chillon"],
    "lima": ["lima", "callao", "miraflores", "huaca pucllana", "huaca huallamarca", "maranga", "cajamarquilla", "puruchuco", "san isidro"],
    "costa_sur": ["ica", "nasca", "nazca", "palpa", "paracas", "pisco", "chincha", "cahuachi", "ocucaje", "arequipa coast", "camana",
                  "moquegua", "ilo", "tacna", "acari", "atico"],
    "sierra_norte": ["cajamarca", "chavin", "huaraz", "ancash", "recuay", "callejon de huaylas", "huari ancash", "kuntur wasi",
                     "pacopampa", "huamachuco", "marcahuamachuco", "otuzco", "carhuaz", "yungay", "caraz", "sihuas"],
    "sierra_central": ["junin", "huancayo", "jauja", "tarma", "huanuco", "kotosh", "pasco", "cerro de pasco", "huancavelica",
                       "ayacucho", "huamanga", "wari", "huari", "conchopata", "apurimac", "abancay", "andahuaylas"],
    "sierra_sur": ["cusco", "cuzco", "machu picchu", "ollantaytambo", "sacsayhuaman", "pisac", "urubamba", "valle sagrado",
                   "chinchero", "arequipa", "colca", "chivay", "tipon", "pikillacta", "raqchi"],
    "altiplano": ["puno", "titicaca", "juliaca", "sillustani", "pucara", "pukara", "tiwanaku", "tiahuanaco", "la paz", "oruro",
                  "altiplano", "chucuito", "ilave"],
    "selva": ["amazonas", "chachapoyas", "kuelap", "kuelap", "loreto", "iquitos", "ucayali", "pucallpa", "madre de dios",
              "puerto maldonado", "san martin", "tarapoto", "moyobamba", "selva", "amazonia", "amazon", "jungle"],
}
GAZETTEER = [(z, name) for z, names in _GAZ.items() for name in names]
GAZETTEER.sort(key=lambda zn: -len(zn[1]))            # longest names first ("huaca de la luna" before "luna")

_COUNTRIES = {"peru": "Peru", "bolivia": "Bolivia", "ecuador": "Ecuador", "chile": "Chile", "colombia": "Colombia",
              "argentina": "Argentina", "mexico": "Mexico", "guatemala": "Guatemala", "egypt": "Egypt", "egipto": "Egypt",
              "china": "China", "japan": "Japan", "japon": "Japan", "grecia": "Greece", "greece": "Greece", "italia": "Italy",
              "italy": "Italy", "iran": "Iran", "irak": "Iraq", "iraq": "Iraq", "turquia": "Turkey", "turkey": "Turkey",
              "india": "India", "nigeria": "Nigeria", "mali": "Mali", "congo": "Congo", "costa rica": "Costa Rica",
              "panama": "Panama", "espana": "Spain", "spain": "Spain", "francia": "France", "france": "France"}
_COUNTRY_REGION = {"Peru": "Andes", "Bolivia": "Andes", "Chile": "Andes", "Argentina": "Andes", "Ecuador": "Northern Andes",
                   "Colombia": "Northern Andes", "Mexico": "Mesoamerica", "Guatemala": "Mesoamerica", "Egypt": "Egypt",
                   "China": "China", "Japan": "Japan", "Greece": "Mediterranean", "Italy": "Mediterranean", "Iran": "Near East",
                   "Iraq": "Near East", "Turkey": "Near East", "India": "South Asia", "Nigeria": "West Africa", "Mali": "West Africa",
                   "Congo": "Central Africa", "Costa Rica": "Isthmo-Colombian", "Panama": "Isthmo-Colombian", "Spain": "Europe",
                   "France": "Europe"}


def _gazetteer(text: str) -> Optional[tuple]:
    f = " " + re.sub(r"[^a-z0-9 ]+", " ", fold(text)) + " "
    for zone, name in GAZETTEER:
        if " %s " % name in f:
            return zone, name
    return None


def _country(text: str) -> Optional[str]:
    f = " " + re.sub(r"[^a-z ]+", " ", fold(text)) + " "
    for k, v in _COUNTRIES.items():
        if " %s " % k in f:
            return v
    return None


def _llm_parse(text: str) -> Optional[dict]:
    """ART's LLM (optional): {"country","zone","site"} from free text in any language."""
    try:
        from yq.macworker.models.llm import chat  # owned by ART; may not exist
    except Exception:
        return None
    prompt = ("Texto de un visitante sobre dónde se encontró un objeto arqueológico: %r\n"
              "Devuelve SOLO JSON: {\"country\": nombre del país en inglés o null, \"zone\": una de %s o null "
              "(solo si es Perú), \"site\": sitio/valle/ciudad o null}." % (text, ZONES))
    try:
        import json
        raw = chat([{"role": "user", "content": prompt}], max_tokens=120, json_mode=True, temperature=0.0)
        d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        zone = d.get("zone") if d.get("zone") in ZONES else None
        return {"country": d.get("country") or None, "zone": zone, "site": d.get("site") or None}
    except Exception:
        return None


def parse_context(ctx: Optional[dict], use_llm: bool = True) -> dict:
    """-> {"given", "country", "zone", "region", "site", "confidence", "source", "text"}."""
    ctx = ctx or {}
    text = " ".join(str(ctx.get(k) or "") for k in ("found_where", "notes")).strip()
    chip = str(ctx.get("region_hint") or "").strip().lower()
    out = {"given": bool(text or (chip and chip != "no_se")), "country": None, "zone": None, "region": None, "site": None,
           "confidence": 0.0, "source": "", "text": text}
    if chip in ZONES:
        out.update(country="Peru", zone=chip, confidence=1.0, source="chip")
    g = _gazetteer(text) if text else None
    if g and not out["zone"]:
        out.update(country="Peru", zone=g[0], site=g[1], confidence=0.9, source="gazetteer")
    elif g:
        out["site"] = g[1]
        if g[0] != out["zone"] and g[0] not in NEIGHBOURS.get(out["zone"], []):
            out["confidence"] = 0.6          # chip and text disagree: trust both less
    if not out["country"] and text:
        c = _country(text)
        if c:
            out.update(country=c, confidence=0.8, source="country")
    if not out["country"] and not out["zone"] and text and use_llm:
        d = _llm_parse(text)
        if d and (d.get("country") or d.get("zone")):
            out.update(country=d.get("country") or ("Peru" if d.get("zone") else None), zone=d.get("zone"),
                       site=d.get("site"), confidence=0.6, source="llm")
    if chip == "otro_pais" and out["country"] in (None, "Peru"):
        out.update(country=out["country"] if out["country"] != "Peru" else None, zone=None,
                   confidence=max(out["confidence"], 0.5), source=out["source"] or "chip")
        out["not_peru"] = True
    if out["country"]:
        out["region"] = _COUNTRY_REGION.get(out["country"]) or match_region(out["country"])
    elif out["zone"]:
        out["region"] = "Andes"
    if not (out["zone"] or out["country"] or out.get("not_peru")):
        out["confidence"] = 0.0
    return out


def compatibility(culture: str, culture_region: Optional[str], hint: dict) -> float:
    """k_c in [-0.5, 1] between a canonical culture and the parsed hint (see module docstring)."""
    if not hint.get("confidence"):
        return 0.0
    zones = CULTURE_ZONES.get(culture)
    if hint.get("not_peru") and not hint.get("country"):
        return -0.5 if culture in PERU_CULTURES else 0.0
    if hint.get("zone") and zones is not None:
        if not zones:
            return 0.0
        if hint["zone"] == zones[0] or (hint["zone"] in zones[:2] and len(zones) <= 3):
            return 1.0
        if hint["zone"] in zones or any(hint["zone"] in NEIGHBOURS.get(z, []) for z in zones[:1]):
            return 0.5
        return 0.0 if culture in ("Inca", "Wari", "Andean Colonial") else -0.5
    region = hint.get("region")
    if region and culture_region:
        if region == culture_region:
            return 0.5 if hint.get("zone") is None else 0.25
        andes = {"Andes", "Northern Andes"}
        if region in andes and culture_region in andes:
            return 0.0
        return -0.5
    return 0.0


def apply_prior(scores: dict, regions: dict, hint: dict, max_boost: float = 1.5, support: float = 0.25) -> tuple:
    """scores: culture -> image-only score (>= 0). regions: culture -> taxonomy region.
    Returns (new normalised scores, {culture: multiplier})."""
    if not scores:
        return {}, {}
    w = float(hint.get("confidence") or 0.0)
    top = max(scores.values())
    mult = {}
    for c, s in scores.items():
        m = max_boost ** (w * compatibility(c, regions.get(c), hint))
        if s < support * top:
            m = min(m, 1.0)
        mult[c] = m
    new = {c: s * mult[c] for c, s in scores.items()}
    tot = sum(new.values()) or 1.0
    return {c: v / tot for c, v in new.items()}, mult


def effect_text(hint: dict, before: Optional[str], after: Optional[str], before_names: dict,
                k_before: Optional[float]) -> str:
    """Simple Spanish sentence for Identification.context_effect_es."""
    if not hint.get("given"):
        return ""
    if not hint.get("confidence"):
        return "No se pudo interpretar el lugar indicado; se usó solo lo que se ve en el objeto."
    name = lambda c: before_names.get(c, c)
    if before and after and before != after:
        return "El lugar ayudó a decidir entre %s y %s." % (name(after), name(before))
    if k_before is not None and k_before < 0:
        return "El lugar no coincide con lo que se ve; se priorizó la imagen."
    if k_before is not None and k_before > 0:
        return "El lugar coincide con lo que se ve; no cambió la respuesta."
    return "El lugar no cambió la respuesta."
