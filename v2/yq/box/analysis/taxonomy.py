"""Controlled vocabularies used by the catalog and the identification.

Museum records spell the same thing in many ways ("Nasca", "Nazca",
"Peru, South Coast, Nasca style (100-600)"; "Moche" / "Mochica"). Here every
free-text record is mapped to a canonical culture, a material class and an
object type so that retrieval votes can be counted and accuracy measured.

Matching is accent- and case-insensitive (see `fold`). Order matters: more
specific entries come first.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional


def fold(text: str) -> str:
    """Lowercase, strip accents: 'Chimú' -> 'chimu'."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower()


# --- Cultures -------------------------------------------------------------------------
# (canonical, spanish, region, [regex aliases on folded text], typical date range)
_CULTURES = [
    # Andes (in depth)
    ("Chavin", "Chavín", "Andes", [r"chavin"], "900-200 BCE"),
    ("Cupisnique", "Cupisnique", "Andes", [r"cupisnique"], "1500-500 BCE"),
    ("Paracas", "Paracas", "Andes", [r"paracas"], "800 BCE-100 CE"),
    ("Nasca", "Nasca", "Andes", [r"\bnasca\b", r"\bnazca\b"], "100 BCE-650 CE"),
    ("Moche", "Mochica", "Andes", [r"\bmoche\b", r"\bmochica\b"], "100-800 CE"),
    ("Recuay", "Recuay", "Andes", [r"recuay"], "100-700 CE"),
    ("Vicus", "Vicús", "Andes", [r"\bvicus\b"], "200 BCE-600 CE"),
    ("Salinar", "Salinar", "Andes", [r"salinar"], "400-100 BCE"),
    ("Gallinazo", "Gallinazo (Virú)", "Andes", [r"gallinazo", r"\bviru\b"], "200 BCE-300 CE"),
    ("Lima", "Lima (cultura)", "Andes", [r"\blima (culture|style)\b", r"maranga"], "100-650 CE"),
    ("Pukara", "Pucará", "Andes", [r"pukara", r"pucara"], "500 BCE-400 CE"),
    ("Wari", "Wari (Huari)", "Andes", [r"\bwari\b", r"\bhuari\b"], "600-1000 CE"),
    ("Tiwanaku", "Tiahuanaco", "Andes", [r"tiwanaku", r"tiahuanaco"], "500-1000 CE"),
    ("Lambayeque", "Lambayeque (Sicán)", "Andes", [r"lambayeque", r"\bsican\b"], "750-1375 CE"),
    ("Chimu", "Chimú", "Andes", [r"\bchimu\b"], "900-1470 CE"),
    ("Chancay", "Chancay", "Andes", [r"chancay"], "1000-1470 CE"),
    ("Ica", "Ica (Chincha)", "Andes", [r"\bica\b(?!\w)", r"chincha"], "1000-1476 CE"),
    ("Inca", "Inca", "Andes", [r"\binca\b", r"\binka\b"], "1400-1533 CE"),
    ("Chiribaya", "Chiribaya", "Andes", [r"chiribaya"], "900-1350 CE"),
    ("Andean Colonial", "Andino colonial (virreinal)", "Andes",
     [r"colonial.*(peru|andes|andean|bolivia)", r"(peru|andes|andean|bolivia).*colonial", r"viceroyalty of peru"], "1533-1821 CE"),
    ("Aymara", "Aimara", "Andes", [r"aymara"], "1400-1900 CE"),
    ("Diaguita", "Diaguita", "Andes", [r"diaguita"], "1000-1536 CE"),
    ("Aguada", "Aguada", "Andes", [r"\baguada\b"], "600-900 CE"),
    ("Mapuche", "Mapuche", "Andes", [r"mapuche"], "1500-1900 CE"),
    # Ecuador, Colombia, Isthmus
    ("Valdivia", "Valdivia", "Northern Andes", [r"valdivia"], "3500-1800 BCE"),
    ("Chorrera", "Chorrera", "Northern Andes", [r"chorrera"], "1300-300 BCE"),
    ("Jama-Coaque", "Jama-Coaque", "Northern Andes", [r"jama.?coaque"], "350 BCE-1530 CE"),
    ("Bahia", "Bahía", "Northern Andes", [r"\bbahia\b"], "500 BCE-500 CE"),
    ("Tumaco-La Tolita", "Tumaco-La Tolita", "Northern Andes", [r"tolita", r"tumaco"], "600 BCE-400 CE"),
    ("Manteno", "Manteño", "Northern Andes", [r"manteno"], "800-1532 CE"),
    ("Capuli", "Capulí", "Northern Andes", [r"capuli", r"narino"], "800-1500 CE"),
    ("Calima", "Calima", "Northern Andes", [r"calima"], "200 BCE-1300 CE"),
    ("Quimbaya", "Quimbaya", "Northern Andes", [r"quimbaya"], "500-1600 CE"),
    ("Muisca", "Muisca", "Northern Andes", [r"muisca"], "600-1600 CE"),
    ("Tairona", "Tairona", "Northern Andes", [r"tairona"], "900-1600 CE"),
    ("Zenu", "Zenú (Sinú)", "Northern Andes", [r"\bzenu\b", r"\bsinu\b"], "200 BCE-1600 CE"),
    ("Cocle", "Coclé", "Isthmo-Colombian", [r"cocle", r"\bconte\b", r"parita", r"macaracas"], "500-1500 CE"),
    ("Chiriqui", "Chiriquí", "Isthmo-Colombian", [r"chiriqui"], "700-1500 CE"),
    ("Diquis", "Diquís", "Isthmo-Colombian", [r"diquis"], "700-1550 CE"),
    ("Veraguas", "Veraguas", "Isthmo-Colombian", [r"veraguas"], "700-1500 CE"),
    ("Guanacaste-Nicoya", "Gran Nicoya", "Isthmo-Colombian", [r"nicoya", r"guanacaste"], "300 BCE-1500 CE"),
    ("Atlantic Watershed", "Vertiente Atlántica (Costa Rica)", "Isthmo-Colombian",
     [r"atlantic watershed", r"linea vieja"], "300 BCE-1500 CE"),
    # Mesoamerica and West Mexico
    ("Olmec", "Olmeca", "Mesoamerica", [r"olmec"], "1500-400 BCE"),
    ("Maya", "Maya", "Mesoamerica", [r"\bmaya\b", r"mayan"], "250 BCE-1500 CE"),
    ("Teotihuacan", "Teotihuacana", "Mesoamerica", [r"teotihuacan"], "100 BCE-650 CE"),
    ("Zapotec", "Zapoteca", "Mesoamerica", [r"zapotec"], "500 BCE-1500 CE"),
    ("Mixtec", "Mixteca", "Mesoamerica", [r"mixtec"], "900-1521 CE"),
    ("Aztec", "Mexica (azteca)", "Mesoamerica", [r"\baztec", r"mexica\b"], "1325-1521 CE"),
    ("Toltec", "Tolteca", "Mesoamerica", [r"toltec"], "900-1150 CE"),
    ("Huastec", "Huasteca", "Mesoamerica", [r"huastec"], "900-1521 CE"),
    ("Veracruz", "Veracruz (Golfo)", "Mesoamerica", [r"veracruz", r"remojadas", r"totonac"], "300 BCE-900 CE"),
    ("Mezcala", "Mezcala", "Mesoamerica", [r"mezcala", r"guerrero"], "700 BCE-1000 CE"),
    ("Tlatilco", "Tlatilco", "Mesoamerica", [r"tlatilco"], "1200-600 BCE"),
    ("Colima", "Colima", "West Mexico", [r"colima"], "300 BCE-300 CE"),
    ("Nayarit", "Nayarit", "West Mexico", [r"nayarit"], "300 BCE-300 CE"),
    ("Jalisco", "Jalisco", "West Mexico", [r"jalisco"], "300 BCE-300 CE"),
    ("Chupicuaro", "Chupícuaro", "West Mexico", [r"chupicuaro"], "500 BCE-100 CE"),
    ("Mexican Colonial", "Novohispano (colonial)", "Mesoamerica", [r"new spain", r"novohispan"], "1521-1821 CE"),
    # North America and Caribbean
    ("American", "Estadounidense (moderna)", "North America", [r"\(american,", r"american, \d{4}", r"united states, \d{4}"], "1700-1950 CE"),
    ("Taino", "Taíno", "Caribbean", [r"taino"], "1000-1500 CE"),
    ("Ancestral Puebloan", "Ancestral Pueblo", "North America",
     [r"ancestral pueblo", r"anasazi", r"mimbres", r"hohokam", r"mogollon"], "700-1500 CE"),
    ("Pueblo", "Pueblo", "North America", [r"\bpueblo\b", r"\bhopi\b", r"\bzuni\b", r"acoma", r"santo domingo", r"san ildefonso"], "1700-1950 CE"),
    ("Mississippian", "Misisipiana", "North America", [r"mississippian"], "800-1600 CE"),
    ("Inuit", "Inuit", "North America", [r"inuit", r"eskimo", r"yupik", r"yup'ik", r"inupiat"], "1000-1950 CE"),
    ("Northwest Coast", "Costa Noroeste", "North America",
     [r"tlingit", r"haida", r"kwakwaka", r"kwakiutl", r"tsimshian", r"nuu-chah", r"northwest coast"], "1750-1950 CE"),
    ("Plains", "Llanuras (Norteamérica)", "North America",
     [r"lakota", r"sioux", r"cheyenne", r"crow\b", r"blackfeet", r"plains"], "1800-1950 CE"),
    ("Amazonian", "Amazónica", "Amazonia", [r"amazon", r"marajo", r"shipibo", r"jivaro", r"shuar", r"karaja", r"kayapo", r"yanomami"], "400-1950 CE"),
    # Africa
    ("Egyptian", "Egipcia (faraónica)", "Egypt",
     [r"egypt(?!.*(islamic|fatimid|mamluk|ottoman))", r"dynasty \d+", r"ptolemaic", r"new kingdom", r"middle kingdom", r"old kingdom", r"predynastic"],
     "3100-30 BCE"),
    ("Coptic", "Copta", "Egypt", [r"coptic", r"byzantine egypt"], "300-800 CE"),
    ("Nubian", "Nubia (Kush)", "Nubia", [r"nubia", r"kush", r"meroit", r"kerma"], "2500 BCE-350 CE"),
    ("Nok", "Nok", "West Africa", [r"\bnok\b"], "900 BCE-300 CE"),
    ("Ife", "Ife", "West Africa", [r"\bife\b"], "1100-1400 CE"),
    ("Benin", "Benín (Edo)", "West Africa", [r"\bedo\b", r"benin kingdom", r"court of benin", r"kingdom of benin"], "1300-1900 CE"),
    ("Yoruba", "Yoruba", "West Africa", [r"yoruba"], "1800-1950 CE"),
    ("Igbo", "Igbo", "West Africa", [r"\bigbo\b"], "1800-1950 CE"),
    ("Akan", "Akan (Asante)", "West Africa", [r"\bakan\b", r"asante", r"ashanti", r"fante"], "1700-1950 CE"),
    ("Baule", "Baulé", "West Africa", [r"baule", r"baoule"], "1800-1950 CE"),
    ("Dogon", "Dogon", "West Africa", [r"dogon"], "1400-1950 CE"),
    ("Bamana", "Bamana", "West Africa", [r"bamana", r"bambara"], "1800-1950 CE"),
    ("Senufo", "Senufo", "West Africa", [r"senufo"], "1800-1950 CE"),
    ("Djenne", "Djenné (Delta interior)", "West Africa", [r"djenne", r"jenne", r"inland niger delta"], "1100-1600 CE"),
    ("Dan", "Dan", "West Africa", [r"\bdan\b(?! [a-z]+ dynasty)"], "1850-1950 CE"),
    ("Mende", "Mende", "West Africa", [r"\bmende\b", r"\bsande\b"], "1850-1950 CE"),
    ("Fang", "Fang", "Central Africa", [r"\bfang\b"], "1800-1950 CE"),
    ("Kongo", "Kongo", "Central Africa", [r"kongo", r"vili\b", r"yombe"], "1700-1950 CE"),
    ("Luba", "Luba", "Central Africa", [r"\bluba\b"], "1800-1950 CE"),
    ("Songye", "Songye", "Central Africa", [r"songye"], "1800-1950 CE"),
    ("Kuba", "Kuba", "Central Africa", [r"\bkuba\b"], "1800-1950 CE"),
    ("Chokwe", "Chokwe", "Central Africa", [r"chokwe"], "1800-1950 CE"),
    ("Kota", "Kota", "Central Africa", [r"\bkota\b"], "1800-1950 CE"),
    ("Cameroon Grassfields", "Pastizales de Camerún", "Central Africa", [r"bamileke", r"bamum", r"bangwa", r"grassfield"], "1800-1950 CE"),
    ("Ethiopian", "Etíope", "East Africa", [r"ethiopia", r"aksum", r"axum"], "300-1950 CE"),
    ("Zulu", "Zulú", "Southern Africa", [r"\bzulu\b", r"\bnguni\b", r"tsonga"], "1800-1950 CE"),
    # Oceania
    ("Maori", "Maorí", "Oceania", [r"maori", r"new zealand"], "1500-1950 CE"),
    ("Hawaiian", "Hawaiana", "Oceania", [r"hawai"], "1700-1900 CE"),
    ("Rapa Nui", "Rapa Nui", "Oceania", [r"rapa nui", r"easter island"], "1200-1900 CE"),
    ("Marquesan", "Marquesana", "Oceania", [r"marquesa"], "1700-1900 CE"),
    ("Asmat", "Asmat", "Oceania", [r"asmat"], "1900-1970 CE"),
    ("Sepik", "Sepik", "Oceania", [r"sepik", r"iatmul", r"abelam", r"sawos"], "1850-1970 CE"),
    ("New Ireland", "Nueva Irlanda", "Oceania", [r"new ireland", r"malagan"], "1850-1950 CE"),
    ("Solomon Islands", "Islas Salomón", "Oceania", [r"solomon island"], "1850-1950 CE"),
    ("Vanuatu", "Vanuatu", "Oceania", [r"vanuatu", r"new hebrides"], "1850-1950 CE"),
    ("Fijian", "Fiyiana", "Oceania", [r"\bfiji"], "1800-1950 CE"),
    ("Aboriginal Australian", "Aborigen australiana", "Oceania", [r"aborigin", r"australia"], "1850-1980 CE"),
    ("Lapita", "Lapita", "Oceania", [r"lapita"], "1500-500 BCE"),
    # Near East and Mediterranean
    ("Sumerian", "Sumeria", "Near East", [r"sumer", r"early dynastic", r"\bur\b"], "3500-2000 BCE"),
    ("Akkadian", "Acadia", "Near East", [r"akkad"], "2350-2150 BCE"),
    ("Babylonian", "Babilonia", "Near East", [r"babylon", r"old babylonian", r"kassite"], "2000-539 BCE"),
    ("Assyrian", "Asiria", "Near East", [r"assyria", r"neo-assyrian"], "1400-600 BCE"),
    ("Elamite", "Elamita", "Near East", [r"elam", r"susa"], "3200-539 BCE"),
    ("Achaemenid", "Aqueménida (Persia)", "Near East", [r"achaemenid"], "550-330 BCE"),
    ("Parthian", "Parta", "Near East", [r"parthian"], "247 BCE-224 CE"),
    ("Sasanian", "Sasánida", "Near East", [r"sasanian", r"sassanian"], "224-651 CE"),
    ("Bactria-Margiana", "Bactria-Margiana", "Near East", [r"bactria", r"margiana", r"oxus"], "2300-1700 BCE"),
    ("Anatolian", "Anatolia antigua", "Near East", [r"hittite", r"anatolia", r"urartian", r"phrygian", r"lydian"], "3000-500 BCE"),
    ("Iranian (ancient)", "Irán antiguo", "Near East", [r"amlash", r"luristan", r"marlik", r"iron age.*iran", r"\biran\b.*bce"], "3000-500 BCE"),
    ("Levantine", "Levante (fenicia, cananea)", "Near East", [r"phoenic", r"canaan", r"levant", r"syro-palest", r"israel"], "3000-300 BCE"),
    ("South Arabian", "Arabia del sur", "Near East", [r"south arabia", r"yemen", r"sabaean"], "800 BCE-300 CE"),
    ("Cypriot", "Chipriota", "Mediterranean", [r"cypr"], "2500-300 BCE"),
    ("Cycladic", "Cicládica", "Mediterranean", [r"cycladic"], "3200-2000 BCE"),
    ("Minoan", "Minoica", "Mediterranean", [r"minoan"], "3000-1100 BCE"),
    ("Mycenaean", "Micénica", "Mediterranean", [r"mycenae", r"helladic"], "1600-1100 BCE"),
    ("Etruscan", "Etrusca", "Mediterranean", [r"etrusc"], "800-100 BCE"),
    ("Greek", "Griega", "Mediterranean",
     [r"\bgreek\b", r"attic\b", r"corinthian", r"boeotian", r"laconian", r"east greek", r"south italian", r"apulian",
      r"campanian", r"lucanian", r"hellenistic", r"geometric"], "900-30 BCE"),
    ("Roman", "Romana", "Mediterranean", [r"\broman\b", r"imperial", r"augustan", r"julio-claudian"], "100 BCE-400 CE"),
    ("Byzantine", "Bizantina", "Mediterranean", [r"byzantin"], "330-1453 CE"),
    ("Celtic", "Celta", "Europe", [r"celtic", r"la tene", r"hallstatt"], "800 BCE-100 CE"),
    ("Viking", "Vikinga", "Europe", [r"viking", r"norse"], "793-1066 CE"),
    ("Merovingian", "Merovingia", "Europe", [r"merovingian", r"frankish", r"langobard", r"lombard", r"visigoth", r"anglo-saxon", r"migration period"], "400-800 CE"),
    ("European Medieval", "Europea medieval", "Europe",
     [r"medieval", r"romanesque", r"gothic", r"carolingian", r"ottonian", r"limoges"], "800-1500 CE"),
    ("European Renaissance", "Europea renacentista", "Europe", [r"renaissance"], "1400-1600 CE"),
    ("European", "Europea (moderna)", "Europe",
     [r"\b(french|german|italian|british|english|dutch|flemish|spanish|austrian|swiss|scandinavian|russian|portuguese)\b", r"meissen", r"sevres", r"delft", r"majolica", r"maiolica"],
     "1500-1900 CE"),
    ("Islamic", "Islámica", "Islamic",
     [r"islamic", r"abbasid", r"umayyad", r"fatimid", r"mamluk", r"ottoman", r"safavid", r"seljuq", r"seljuk", r"timurid", r"ilkhanid", r"nasrid", r"qajar", r"samanid", r"ayyubid", r"iznik", r"kashan", r"nishapur", r"mughal"],
     "650-1900 CE"),
    # Asia
    ("Chinese Neolithic", "China neolítica", "China", [r"neolithic.*china", r"yangshao", r"majiayao", r"longshan", r"liangzhu", r"hongshan", r"banshan", r"machang"], "5000-2000 BCE"),
    ("Chinese Shang-Zhou", "China Shang-Zhou (bronce)", "China", [r"\bshang\b", r"zhou dynasty", r"western zhou", r"eastern zhou", r"warring states", r"spring and autumn"], "1600-221 BCE"),
    ("Chinese Han", "China Han", "China", [r"\bhan dynasty", r"\bqin dynasty", r"\bhan\b\s*\(", r"eastern han", r"western han"], "221 BCE-220 CE"),
    ("Chinese Tang", "China Tang", "China", [r"tang dynasty", r"\btang\b\s*\(", r"sui dynasty", r"northern wei", r"six dynasties", r"northern qi", r"jin dynasty \(265"], "220-907 CE"),
    ("Chinese Song-Yuan", "China Song-Yuan", "China", [r"song dynasty", r"yuan dynasty", r"liao dynasty", r"jin dynasty \(1115", r"northern song", r"southern song"], "960-1368 CE"),
    ("Chinese Ming-Qing", "China Ming-Qing", "China", [r"ming dynasty", r"qing dynasty", r"kangxi", r"qianlong", r"yongzheng", r"jingdezhen"], "1368-1912 CE"),
    ("Chinese", "China", "China", [r"\bchina\b", r"chinese"], "2000 BCE-1912 CE"),
    ("Japanese Jomon-Kofun", "Japón Jōmon-Kofun", "Japan", [r"jomon", r"yayoi", r"kofun", r"haniwa"], "10000 BCE-538 CE"),
    ("Japanese", "Japón", "Japan", [r"japan", r"edo period", r"meiji", r"momoyama", r"muromachi", r"kamakura", r"heian", r"arita", r"imari"], "538-1912 CE"),
    ("Korean", "Corea", "Korea", [r"korea", r"goryeo", r"koryo", r"joseon", r"choson", r"silla", r"baekje", r"gaya"], "57 BCE-1910 CE"),
    ("Gandharan", "Gandhara", "South Asia", [r"gandhara", r"kushan"], "100 BCE-500 CE"),
    ("Indus", "Harappa (Indo)", "South Asia", [r"indus valley", r"harappa", r"mohenjo"], "2600-1900 BCE"),
    ("Indian", "India", "South Asia", [r"\bindia\b", r"indian", r"chola", r"gupta", r"pala\b", r"vijayanagara", r"pakistan", r"bangladesh", r"sri lanka"], "300 BCE-1900 CE"),
    ("Himalayan", "Himalaya (Tíbet, Nepal)", "South Asia", [r"tibet", r"nepal", r"bhutan"], "700-1900 CE"),
    ("Khmer", "Jemer (Camboya)", "Southeast Asia", [r"khmer", r"cambodia", r"angkor"], "600-1400 CE"),
    ("Thai", "Tailandia", "Southeast Asia", [r"thailand", r"\bthai\b", r"ban chiang", r"sukhothai", r"sawankhalok", r"dvaravati"], "1000 BCE-1900 CE"),
    ("Vietnamese", "Vietnam", "Southeast Asia", [r"vietnam", r"dong son", r"champa", r"\bcham\b"], "500 BCE-1900 CE"),
    ("Indonesian", "Indonesia", "Southeast Asia", [r"indonesia", r"\bjava", r"sumatra", r"borneo", r"bali\b", r"dayak", r"batak", r"philippine", r"myanmar", r"burma"], "700-1950 CE"),
    ("Central Asian", "Asia central", "Central Asia", [r"central asia", r"sogdian", r"scythian", r"steppe", r"ordos", r"afghanistan", r"uzbekistan", r"turkmen"], "1000 BCE-1500 CE"),
]

CULTURES = [{"name": c[0], "name_es": c[1], "region": c[2], "patterns": [re.compile(p) for p in c[3]], "dates": c[4]}
            for c in _CULTURES]
CULTURE_BY_NAME = {c["name"]: c for c in CULTURES}

# Region fallback from geography words when no culture matched (region only).
_REGION_FALLBACK = [
    ("Andes", [r"\bperu\b", r"bolivia", r"\bchile\b", r"andes", r"andean", r"argentina"]),
    ("Northern Andes", [r"ecuador", r"colombia"]),
    ("Isthmo-Colombian", [r"costa rica", r"panama", r"nicaragua"]),
    ("Mesoamerica", [r"mexico", r"guatemala", r"honduras", r"el salvador", r"belize", r"mesoamerica"]),
    ("North America", [r"united states", r"canada", r"native american"]),
    ("Oceania", [r"papua", r"new guinea", r"melanesia", r"polynesia", r"micronesia", r"oceania"]),
    ("West Africa", [r"nigeria", r"ghana", r"mali\b", r"cote d'ivoire", r"ivory coast", r"burkina", r"sierra leone", r"liberia", r"guinea\b", r"togo", r"senegal"]),
    ("Central Africa", [r"congo", r"gabon", r"cameroon", r"angola", r"zaire"]),
    ("East Africa", [r"kenya", r"tanzania", r"somalia", r"uganda", r"sudan"]),
    ("Southern Africa", [r"south africa", r"zimbabwe", r"mozambique", r"zambia"]),
    ("Near East", [r"iraq", r"mesopotamia", r"syria", r"turkey", r"anatolia", r"iran", r"jordan", r"lebanon"]),
    ("Mediterranean", [r"greece", r"italy", r"crete", r"rhodes", r"cyprus"]),
]
_REGION_FALLBACK = [(r, [re.compile(p) for p in ps]) for r, ps in _REGION_FALLBACK]

REGIONS_ES = {
    "Andes": "Andes centrales (Perú, Bolivia, norte de Chile)", "Northern Andes": "Andes del norte (Ecuador, Colombia)",
    "Isthmo-Colombian": "Área istmo-colombiana", "Mesoamerica": "Mesoamérica", "West Mexico": "Occidente de México",
    "North America": "Norteamérica", "Caribbean": "Caribe", "Amazonia": "Amazonía", "Egypt": "Egipto", "Nubia": "Nubia",
    "West Africa": "África occidental", "Central Africa": "África central", "East Africa": "África oriental",
    "Southern Africa": "África austral", "Oceania": "Oceanía", "Near East": "Cercano Oriente", "Mediterranean": "Mediterráneo",
    "Europe": "Europa", "Islamic": "Mundo islámico", "China": "China", "Japan": "Japón", "Korea": "Corea",
    "South Asia": "Asia del sur", "Southeast Asia": "Sudeste asiático", "Central Asia": "Asia central",
}


def match_culture(*texts: str) -> Optional[dict]:
    """First canonical culture whose pattern matches any of `texts` (checked in text order)."""
    for t in texts:
        f = fold(t)
        if not f:
            continue
        best = None
        for c in CULTURES:
            for p in c["patterns"]:
                hit = p.search(f)
                if hit and (best is None or hit.start() < best[0]):
                    best = (hit.start(), c)
        if best:
            return best[1]
    return None


def match_region(*texts: str) -> Optional[str]:
    c = match_culture(*texts)
    if c:
        return c["region"]
    for t in texts:
        f = fold(t)
        for region, pats in _REGION_FALLBACK:
            if any(p.search(f) for p in pats):
                return region
    return None


# --- Materials ---------------------------------------------------------------------------
# (fine material, class, spanish, regex aliases). Matched against the medium text; the
# material mentioned FIRST in the text wins (museums list the main material first).
_MATERIALS = [
    ("faience", "glass_faience", "fayenza egipcia", [r"faience", r"frit\b", r"egyptian blue"]),
    ("porcelain", "ceramic", "porcelana", [r"porcelain", r"celadon"]),
    ("stoneware", "ceramic", "gres cerámico", [r"stoneware"]),
    ("earthenware", "ceramic", "cerámica (barro cocido)", [r"earthenware", r"terracotta", r"terra cotta", r"ceramic",
                                                        r"pottery", r"\bclay\b", r"slip\b", r"slip-painted", r"tin-glazed", r"fritware"]),
    ("gold", "metal", "oro", [r"\bgold\b", r"gold alloy", r"tumbaga", r"electrum"]),
    ("silver", "metal", "plata", [r"\bsilver\b"]),
    ("copper alloy", "metal", "bronce / aleación de cobre", [r"bronze", r"copper", r"brass", r"arsenical"]),
    ("iron", "metal", "hierro", [r"\biron\b", r"\bsteel\b"]),
    ("lead", "metal", "plomo", [r"\blead\b", r"pewter", r"\btin\b"]),
    ("jade", "stone", "jade / piedra verde", [r"\bjade", r"jadeite", r"nephrite", r"greenstone", r"serpentine"]),
    ("obsidian", "stone", "obsidiana", [r"obsidian"]),
    ("marble", "stone", "mármol", [r"marble"]),
    ("limestone", "stone", "piedra caliza", [r"limestone", r"travertine"]),
    ("alabaster", "stone", "alabastro / calcita", [r"alabaster", r"calcite", r"gypsum", r"huamanga"]),
    ("steatite", "stone", "esteatita", [r"steatite", r"soapstone", r"chlorite"]),
    ("basalt", "stone", "basalto / granito", [r"basalt", r"granite", r"diorite", r"andesite", r"gabbro", r"granodiorite"]),
    ("sandstone", "stone", "arenisca", [r"sandstone", r"schist", r"slate", r"phyllite"]),
    ("gemstone", "stone", "piedra semipreciosa", [r"turquoise", r"lapis", r"carnelian", r"quartz", r"crystal", r"chrysocolla",
                                                  r"sodalite", r"amethyst", r"agate", r"chalcedony", r"garnet", r"malachite", r"jasper"]),
    ("stone", "stone", "piedra", [r"\bstone\b", r"\brock\b"]),
    ("glass", "glass_faience", "vidrio", [r"\bglass\b"]),
    ("bone", "bone_ivory_shell", "hueso", [r"\bbone\b", r"antler", r"\bhorn\b", r"tooth", r"teeth", r"tusk"]),
    ("ivory", "bone_ivory_shell", "marfil", [r"ivory"]),
    ("shell", "bone_ivory_shell", "concha (Spondylus, nácar)", [r"\bshell", r"spondylus", r"mother-of-pearl", r"nacre", r"conch", r"coral"]),
    ("wood", "wood", "madera", [r"\bwood", r"bamboo"]),
    ("textile", "textile", "textil (algodón, fibra de camélido, lana)", [r"cotton", r"camelid", r"alpaca", r"vicuna", r"\bwool", r"\bsilk\b", r"linen",
                                                                        r"textile", r"tapestry", r"\bfiber\b", r"yarn", r"felt\b", r"velvet"]),
    ("feather", "other_organic", "plumas", [r"feather"]),
    ("plant fiber", "other_organic", "fibra vegetal / cestería", [r"basket", r"rattan", r"reed", r"raffia", r"grass", r"palm", r"gourd", r"bark", r"straw"]),
    ("leather", "other_organic", "cuero / piel", [r"leather", r"hide\b", r"rawhide", r"skin\b", r"fur\b"]),
    ("lacquer", "other_organic", "laca", [r"lacquer"]),
    ("paper", "other", "papel", [r"\bpaper\b", r"papyrus", r"parchment", r"vellum"]),
]
MATERIALS = [{"name": m[0], "cls": m[1], "name_es": m[2], "patterns": [re.compile(p) for p in m[3]]} for m in _MATERIALS]
MATERIAL_BY_NAME = {m["name"]: m for m in MATERIALS}
MATERIAL_CLASSES_ES = {
    "ceramic": "cerámica", "metal": "metal", "stone": "piedra", "wood": "madera", "textile": "textil",
    "bone_ivory_shell": "hueso, marfil o concha", "glass_faience": "vidrio o fayenza", "other_organic": "material orgánico",
    "other": "otro",
}
# Typical densities (g/cm3) per material class, used to compare with the apparent density.
DENSITY_G_CM3 = {
    "ceramic": (1.6, 2.5), "stone": (2.3, 3.4), "metal": (7.5, 19.3), "wood": (0.35, 1.2), "textile": (0.1, 1.5),
    "bone_ivory_shell": (1.6, 2.9), "glass_faience": (1.8, 2.8), "other_organic": (0.1, 1.3), "other": (0.1, 20.0),
}
DENSITY_FINE_G_CM3 = {"gold": (15.0, 19.3), "silver": (9.5, 10.5), "copper alloy": (7.5, 8.9), "iron": (7.0, 7.9),
                      "lead": (7.3, 11.3), "jade": (2.6, 3.4), "obsidian": (2.3, 2.6), "marble": (2.5, 2.8)}


def match_material(*texts: str) -> Optional[dict]:
    """Material mentioned earliest in the first non-empty text that mentions any material."""
    for t in texts:
        f = fold(t)
        if not f:
            continue
        best = None
        for m in MATERIALS:
            for p in m["patterns"]:
                hit = p.search(f)
                if hit and (best is None or hit.start() < best[0]):
                    best = (hit.start(), m)
        if best:
            return best[1]
    return None


# --- Object types ------------------------------------------------------------------------
_TYPES = [
    ("stirrup-spout vessel", "vessel", "botella de asa estribo", [r"stirrup"]),
    ("double-spout vessel", "vessel", "botella de doble pico y asa puente", [r"double.?spout", r"bridge"]),
    ("portrait vessel", "vessel", "vasija retrato", [r"portrait (head )?vessel", r"portrait bottle", r"head vessel"]),
    ("effigy vessel", "vessel", "vasija escultórica (efigie)", [r"effigy (vessel|jar|bottle|pot)", r"figural (vessel|bottle|jar)"]),
    ("aryballos", "vessel", "aríbalo / urpu", [r"aryball", r"\burpu\b"]),
    ("kero", "vessel", "kero / vaso", [r"\bkero\b", r"\bqero\b", r"beaker", r"tumbler"]),
    ("amphora", "vessel", "ánfora", [r"amphora"]),
    ("mask", "mask", "máscara", [r"\bmask"]),
    ("figurine", "figure", "figurina", [r"figurine", r"statuette", r"ushabti", r"shabti", r"amulet"]),
    ("ear ornament", "ornament", "orejera", [r"ear ?(ornament|spool|flare|plug)", r"earspool", r"earring"]),
    ("nose ornament", "ornament", "narigera", [r"nose ornament", r"nose ring"]),
    ("jewelry", "ornament", "joya / adorno", [r"pendant", r"necklace", r"bead", r"bracelet", r"ring\b", r"pectoral", r"brooch",
                                              r"fibula", r"pin\b", r"tupu", r"ornament", r"plaque", r"diadem", r"crown", r"headdress", r"jewel"]),
    ("tunic", "textile", "unku / túnica", [r"tunic", r"\bunku\b", r"poncho", r"garment", r"mantle", r"shawl", r"robe", r"shirt"]),
    ("textile", "textile", "textil / tejido", [r"textile", r"cloth", r"band\b", r"border", r"bag\b", r"chuspa", r"sash", r"belt",
                                              r"tapestry", r"hanging", r"carpet", r"rug\b"]),
    ("weapon", "tool", "arma", [r"mace", r"club", r"axe\b", r"\bax\b", r"sword", r"dagger", r"spear", r"knife", r"blade", r"arrow", r"projectile",
                                r"point\b", r"helmet", r"shield", r"armor", r"armour", r"tumi"]),
    ("tool", "tool", "herramienta", [r"tool", r"spindle", r"whorl", r"needle", r"chisel", r"adze", r"celt\b", r"scraper", r"spatula",
                                      r"spoon", r"ladle", r"lime", r"mortar", r"pestle", r"metate", r"stamp", r"seal", r"roller", r"weight"]),
    ("musical instrument", "instrument", "instrumento musical", [r"whistle", r"flute", r"panpipe", r"ocarina", r"drum", r"rattle", r"trumpet",
                                                                 r"bell", r"instrument", r"antara", r"quena"]),
    ("relief", "relief", "relieve / placa", [r"relief", r"stela", r"tile", r"tablet", r"architectural", r"frieze", r"panel"]),
    ("coin", "coin", "moneda / medalla", [r"\bcoin", r"medal", r"stater", r"drachm", r"tetradrachm", r"denarius", r"aureus"]),
    ("mirror", "other", "espejo", [r"mirror"]),
    ("model", "figure", "maqueta / modelo", [r"\bmodel\b"]),
    ("bottle", "vessel", "botella", [r"bottle", r"flask", r"ewer", r"pitcher", r"jug\b", r"lekythos", r"oinochoe", r"alabastron", r"unguentarium"]),
    ("jar", "vessel", "cántaro / jarra", [r"\bjar\b", r"\bjars\b", r"olla", r"\bpot\b", r"urn", r"hydria", r"pithos", r"krater", r"pyxis", r"jardiniere"]),
    ("bowl", "vessel", "cuenco", [r"\bbowl", r"\bdish\b", r"\bplate\b", r"\bplatter\b", r"basin", r"tazza", r"kylix", r"phiale", r"saucer"]),
    ("cup", "vessel", "vaso / copa", [r"\bcup\b", r"\bcups\b", r"goblet", r"chalice", r"kantharos", r"skyphos", r"\bmug\b", r"teabowl", r"tea bowl"]),
    ("vessel", "vessel", "vasija", [r"vessel", r"\bvase\b", r"container", r"censer", r"incense burner", r"\blamp\b", r"teapot", r"box\b", r"canister"]),
    ("figure", "figure", "escultura / figura", [r"figure", r"sculpture", r"statue", r"\bhead\b", r"\bbust\b", r"torso", r"idol", r"deity",
                                               r"buddha", r"bodhisattva", r"effigy"]),
]
OBJECT_TYPES = [{"name": t[0], "cls": t[1], "name_es": t[2], "patterns": [re.compile(p) for p in t[3]]} for t in _TYPES]
OBJECT_TYPE_BY_NAME = {t["name"]: t for t in OBJECT_TYPES}
TYPE_CLASSES_ES = {"vessel": "vasija / recipiente", "figure": "figura / escultura", "mask": "máscara", "ornament": "adorno / joya",
                   "textile": "textil", "tool": "herramienta / arma", "instrument": "instrumento musical", "relief": "relieve / placa",
                   "coin": "moneda / medalla", "other": "otro"}


def match_type(*texts: str) -> Optional[dict]:
    """Texts in priority order; within a text, the first type in list order (most specific first)."""
    for t in texts:
        f = fold(t)
        if not f:
            continue
        for ty in OBJECT_TYPES:
            if any(p.search(f) for p in ty["patterns"]):
                return ty
    return None


# --- Records ------------------------------------------------------------------------------
# 2D works that the box can never contain; they are left out of the catalog.
EXCLUDE_CLASS = re.compile(r"paint|print|drawing|photograph|book|manuscript|calligraph|album|woodblock|ephemera|negative|poster|"
                           r"watercolou?r|miniature painting|architectural drawing|folio|scroll|screen\b|wallpaper|furniture")


def excluded_classification(*texts: str) -> bool:
    return any(EXCLUDE_CLASS.search(fold(t)) for t in texts if t)


def normalize_record(rec: dict) -> dict:
    """Add canonical fields (culture_norm, region_norm, material_norm, material_cls, type_norm, type_cls)
    to a raw catalog record with keys title, culture, period, date, medium, classification, object_name,
    geography, department."""
    culture_texts = [rec.get("culture", ""), rec.get("period", ""), rec.get("geography", ""), rec.get("title", ""),
                     rec.get("department", "")]
    c = match_culture(*culture_texts)
    region = c["region"] if c else match_region(*culture_texts)
    m = match_material(rec.get("medium", ""), rec.get("classification", ""), rec.get("title", ""))
    if rec.get("source") in ("aic", "cma"):      # their "type" is a broad class: the title is more specific
        t = match_type(rec.get("title", ""), rec.get("object_name", ""), rec.get("classification", ""))
    else:                                        # The Met's objectName is curated and specific
        t = match_type(rec.get("object_name", ""), rec.get("title", ""), rec.get("classification", ""))
    rec["culture_norm"] = c["name"] if c else None
    rec["region_norm"] = region
    rec["material_norm"] = m["name"] if m else None
    rec["material_cls"] = m["cls"] if m else None
    rec["type_norm"] = t["name"] if t else None
    rec["type_cls"] = t["cls"] if t else None
    return rec


_YEAR = re.compile(r"(-?\d{1,5})")


def format_period(begin: Optional[int], end: Optional[int]) -> str:
    """-200, 100 -> '200 a. C.-100 d. C.'"""
    def one(y):
        return "%d a. C." % (-y) if y < 0 else "%d d. C." % y
    if begin is None and end is None:
        return ""
    if begin is None or end is None or begin == end:
        return one(begin if begin is not None else end)
    if (begin < 0) == (end < 0):
        return "%d-%s" % (abs(begin), one(end))
    return "%s-%s" % (one(begin), one(end))
