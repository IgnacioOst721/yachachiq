"""Build yq/common/data/languages.json: every spoken language Yachachiq can hear,
speak or translate, with honest per-engine availability.

Run (needs internet once; the robot itself never runs this):

    cd ~/yachachiq/v2
    .venvs/voice/bin/python tools/voice_build_languages.py

Sources (all downloaded into ~/yq-data/cache/voice_sources and listed in the
output file's "meta" block):
  - ISO 639-3 code tables (SIL, the registration authority)
  - Glottolog CLDF languages.csv (macro-area and countries, CC-BY-4.0)
  - Whisper's LANGUAGES dict (openai/whisper tokenizer.py, MIT) and its
    per-language FLEURS WER chart for large-v3 (language-breakdown.svg)
  - Omnilingual ASR supported_langs + per-language CER of the 7B LLM model
    (facebookresearch/omnilingual-asr, Apache-2.0)
  - MMS-TTS model list (Hugging Face API, author=facebook, mms-tts-*)
  - MMS-LID-4017 label list (facebook/mms-lid-4017 config.json)
  - NLLB-200 language codes (transformers tokenization_nllb.py)
  - MADLAD-400 <2xx> target tags (the model's SentencePiece vocabulary)
  - Piper voices.json (rhasspy/piper-voices)
  - CLDR language names in Spanish, English and the language itself (babel)
Peruvian languages get hand-checked Spanish names (Ministerio de Cultura,
BDPI) and are verified against the ISO table at build time.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

from yq.common import config  # noqa: E402

CACHE = config.DATA_DIR / "cache" / "voice_sources"
OUT = V2 / "yq" / "common" / "data" / "languages.json"

SOURCES = {
    "iso6393": "https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab",
    "iso6393_macro": "https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3-macrolanguages.tab",
    "glottolog": "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv",
    "whisper_tokenizer": "https://raw.githubusercontent.com/openai/whisper/main/whisper/tokenizer.py",
    "whisper_breakdown": "https://raw.githubusercontent.com/openai/whisper/main/language-breakdown.svg",
    "omni_langs": "https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py",
    "omni_cer": "https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/main/per_language_results_table_7B_llm_asr.csv",
    "mms_lid": "https://huggingface.co/facebook/mms-lid-4017/resolve/main/config.json",
    "nllb_tokenizer": "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/nllb/tokenization_nllb.py",
    "madlad_spm": "https://huggingface.co/google/madlad400-3b-mt/resolve/main/spiece.model",
    "piper_voices": "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json",
    "mms_tts_api": "https://huggingface.co/api/models?author=facebook&search=mms-tts-&limit=1000",
}

# ---------------------------------------------------------------------------
# Hand-curated facts
# ---------------------------------------------------------------------------
# Whisper two-letter code -> canonical code. Macrolanguages resolve to the
# individual language the other engines (NLLB, Omnilingual) use.
WHISPER_TO_CANON_OVERRIDE = {
    "zh": "cmn_Hans", "yue": "yue_Hant", "ar": "arb_Arab", "fa": "pes_Arab", "ms": "zsm_Latn",
    "sw": "swh_Latn", "et": "ekk_Latn", "lv": "lvs_Latn", "uz": "uzn_Latn", "az": "azj_Latn",
    "no": "nob_Latn", "nn": "nno_Latn", "sq": "als_Latn", "mn": "khk_Cyrl", "ps": "pbt_Arab",
    "ne": "npi_Deva", "yi": "ydd_Hebr", "jw": "jav_Latn", "sr": "srp_Cyrl", "tl": "tgl_Latn",
    "haw": "haw_Latn", "pa": "pan_Guru", "sd": "snd_Arab", "bo": "bod_Tibt", "sa": "san_Deva",
    "la": "lat_Latn", "mg": "plt_Latn", "om": "gaz_Latn",
}

# Default script when no engine tells us (rare).
DEFAULT_SCRIPT = {
    "Arab": {"urd", "pnb", "skr", "ckb", "uig", "prs"},
    "Cyrl": {"rus", "ukr", "bel", "bul", "mkd", "kaz", "kir", "tgk", "tat", "bak", "chv", "sah", "tuk"},
}

# Peru: iso3 -> (name_es, name_native, extra search aliases). Names follow the
# Ministerio de Cultura (BDPI) usage; the ISO code is checked against SIL.
PERU = [
    ("spa", "Español (castellano)", "Español", ["castellano", "spanish"]),
    ("quy", "Quechua ayacuchano (chanka)", "Runasimi (chanka)",
     ["quechua", "runasimi", "runa simi", "chanka", "chanca", "ayacucho", "apurimac", "huancavelica"]),
    ("quz", "Quechua cusqueño (collao)", "Runasimi (qusqu-qullaw)",
     ["quechua", "runasimi", "runa simi", "cusco", "cuzco", "qusqu", "collao", "qullaw"]),
    ("qxp", "Quechua puneño (collao)", "Runasimi (qullaw)", ["quechua", "runasimi", "puno", "collao", "qullaw"]),
    ("ayr", "Aimara", "Aymar aru", ["aymara", "aimara", "jaqi aru", "puno"]),
    ("qwh", "Quechua de Áncash (Huaylas)", "Runashimi (Huaylas)", ["quechua", "ancash", "huaylas", "huaraz"]),
    ("qxn", "Quechua de Áncash (Conchucos norte)", "Runashimi (Conchucos)", ["quechua", "ancash", "conchucos"]),
    ("qxo", "Quechua de Áncash (Conchucos sur)", "Runashimi (Conchucos)", ["quechua", "ancash", "conchucos"]),
    ("qvw", "Quechua huanca (Huaylla)", "Wanka shimi", ["quechua", "huanca", "wanka", "junin", "huancayo"]),
    ("qxw", "Quechua huanca (Jauja)", "Wanka shimi", ["quechua", "huanca", "wanka", "jauja", "junin"]),
    ("qvn", "Quechua de Junín norte", "Runashimi (Junín)", ["quechua", "junin", "tarma"]),
    ("qvh", "Quechua de Huánuco (Huamalíes-Dos de Mayo)", "Runashimi (Huánuco)", ["quechua", "huanuco"]),
    ("qub", "Quechua de Huánuco (Huallaga)", "Runashimi (Huánuco)", ["quechua", "huanuco", "huallaga"]),
    ("qxh", "Quechua de Huánuco (Panao)", "Runashimi (Panao)", ["quechua", "huanuco", "panao"]),
    ("qvm", "Quechua de Huánuco (Margos-Yarowilca-Lauricocha)", "Runashimi (Margos)", ["quechua", "huanuco"]),
    ("qur", "Quechua de Pasco (Yanahuanca)", "Runashimi (Yanahuanca)", ["quechua", "pasco"]),
    ("qxt", "Quechua de Pasco (Santa Ana de Tusi)", "Runashimi (Tusi)", ["quechua", "pasco"]),
    ("qvc", "Quechua de Cajamarca", "Runashimi (Cajamarca)", ["quechua", "cajamarca"]),
    ("quf", "Quechua de Lambayeque (Inkawasi-Kañaris)", "Runashimi (Inkawasi)",
     ["quechua", "lambayeque", "inkawasi", "incahuasi", "kanaris", "cañaris"]),
    ("qvs", "Quechua de San Martín (Lamas)", "Kichwa (Lamas)", ["quechua", "kichwa", "lamas", "san martin"]),
    ("quk", "Quechua de Chachapoyas", "Runashimi (Chachapoyas)", ["quechua", "chachapoyas", "amazonas"]),
    ("qve", "Quechua de Apurímac oriental", "Runasimi (Apurímac)", ["quechua", "apurimac"]),
    ("qvl", "Quechua de Lima (Cajatambo)", "Runashimi (Cajatambo)", ["quechua", "lima", "cajatambo"]),
    ("qvp", "Quechua de Lima (Pacaraos)", "Runashimi (Pacaraos)", ["quechua", "lima", "pacaraos"]),
    ("qux", "Quechua de Lima (Yauyos)", "Runashimi (Yauyos)", ["quechua", "lima", "yauyos"]),
    ("qxa", "Quechua de Áncash (Chiquián)", "Runashimi (Chiquián)", ["quechua", "ancash", "chiquian"]),
    ("qwa", "Quechua de Áncash (Corongo)", "Runashimi (Corongo)", ["quechua", "ancash", "corongo"]),
    ("qws", "Quechua de Áncash (Sihuas)", "Runashimi (Sihuas)", ["quechua", "ancash", "sihuas"]),
    ("qxu", "Quechua de Arequipa (La Unión)", "Runasimi (La Unión)", ["quechua", "arequipa"]),
    ("qvo", "Kichwa del Napo", "Kichwa (Napo)", ["kichwa", "quichua", "napo", "loreto"]),
    ("qvz", "Kichwa del Pastaza norte", "Kichwa (Pastaza)", ["kichwa", "quichua", "pastaza", "loreto"]),
    ("qup", "Kichwa del Pastaza sur", "Kichwa (Pastaza)", ["kichwa", "quichua", "pastaza", "inga", "loreto"]),
    ("jqr", "Jaqaru", "Jaqaru", ["jaqaru", "kawki", "cauqui", "tupe", "yauyos"]),
    ("ayc", "Aimara del sur", "Aymar aru (sur)", ["aymara", "aimara"]),
    # Amazonian languages (BDPI names; alternative/older names as aliases)
    ("cni", "Asháninka", "Asháninka", ["ashaninka", "campa", "ene", "tambo", "junin"]),
    ("agr", "Awajún", "Awajún", ["awajun", "aguaruna", "amazonas"]),
    ("shp", "Shipibo-Konibo", "Shipibo-Konibo", ["shipibo", "konibo", "conibo", "ucayali"]),
    ("cbt", "Shawi", "Shawi", ["shawi", "chayahuita", "chayawita", "loreto"]),
    ("mcb", "Matsigenka", "Matsigenka", ["matsigenka", "machiguenga", "machiguenga", "cusco"]),
    ("prq", "Ashéninka (Perené)", "Ashéninka", ["asheninka", "perene", "campa"]),
    ("cjo", "Ashéninka (Pajonal)", "Ashéninka", ["asheninka", "pajonal", "campa"]),
    ("cpu", "Ashéninka (Pichis)", "Ashéninka", ["asheninka", "pichis", "campa"]),
    ("cpc", "Ashéninka (Apurucayali)", "Ajyíninka", ["asheninka", "ajyininka", "apurucayali"]),
    ("cpb", "Ashéninka (Ucayali-Yurúa)", "Ashéninka", ["asheninka", "ucayali", "yurua"]),
    ("hub", "Wampis", "Wampis", ["wampis", "huambisa"]),
    ("yad", "Yagua", "Yagua", ["yagua", "yahua", "loreto"]),
    ("ame", "Yanesha", "Yanesha'", ["yanesha", "amuesha", "amuesha"]),
    ("tca", "Tikuna", "Tikuna", ["tikuna", "ticuna", "magüta"]),
    ("acu", "Achuar", "Achuar", ["achuar", "achual", "shiwiar"]),
    ("cbu", "Kandozi-Chapra", "Kandozi", ["kandozi", "candoshi", "chapra", "shapra"]),
    ("cod", "Kukama-Kukamiria", "Kukama", ["kukama", "cocama", "cocamilla"]),
    ("cbr", "Kakataibo", "Kakataibo", ["kakataibo", "cashibo", "cacataibo"]),
    ("ura", "Urarina", "Urarina", ["urarina", "loreto"]),
    ("pib", "Yine", "Yine", ["yine", "piro"]),
    ("boa", "Bora", "Bora", ["bora", "loreto"]),
    ("hug", "Harakbut (Huachipaeri)", "Harakbut", ["harakbut", "huachipaeri", "madre de dios"]),
    ("amr", "Harakbut (Amarakaeri)", "Harakbut", ["harakbut", "amarakaeri", "madre de dios"]),
    ("cbs", "Cashinahua", "Hantxa kuin", ["cashinahua", "kaxinawa", "huni kuin"]),
    ("kaq", "Capanahua", "Capanahua", ["capanahua", "kapanawa"]),
    ("cot", "Kakinte", "Kakinte", ["kakinte", "caquinte"]),
    ("not", "Nomatsigenga", "Nomatsigenga", ["nomatsigenga", "nomatsiguenga"]),
    ("cox", "Nanti", "Nanti", ["nanti"]),
    ("mcf", "Matsés", "Matsés", ["matses", "mayoruna"]),
    ("ese", "Ese Eja", "Ese Eja", ["ese eja", "huarayo", "madre de dios"]),
    ("sey", "Secoya", "Secoya", ["secoya", "airo pai"]),
    ("mcd", "Sharanahua", "Sharanahua", ["sharanahua"]),
    ("yaa", "Yaminahua", "Yaminahua", ["yaminahua", "yaminawa"]),
    ("mts", "Yora (nahua)", "Yora", ["yora", "nahua"]),
    ("amc", "Amahuaca", "Amahuaca", ["amahuaca", "amawaka"]),
    ("cul", "Madija (culina)", "Madija", ["madija", "culina", "kulina"]),
    ("huu", "Murui-Muinanɨ", "Murui", ["murui", "huitoto", "witoto"]),
    ("ore", "Maijɨki (orejón)", "Maijɨki", ["maijiki", "orejon"]),
    ("oca", "Ocaina", "Ocaina", ["ocaina"]),
    ("arl", "Arabela", "Arabela", ["arabela"]),
    ("jeb", "Shiwilu", "Shiwilu", ["shiwilu", "jebero"]),
    ("iqu", "Iquitu", "Iquitu", ["iquitu", "iquito"]),
    ("isc", "Iskonawa", "Iskonawa", ["iskonawa", "isconahua"]),
    ("omg", "Omagua", "Omagua", ["omagua"]),
    ("ccc", "Chamicuro", "Chamicuro", ["chamicuro"]),
    ("inp", "Iñapari", "Iñapari", ["inapari", "iñapari"]),
    ("myr", "Muniche", "Muniche", ["muniche"]),
    ("rgr", "Resígaro", "Resígaro", ["resigaro"]),
    ("trr", "Taushiro", "Taushiro", ["taushiro"]),
]

# World's most spoken languages (approximate total-speaker order, Ethnologue
# top-list style), used for `popular` after Spanish and the Peruvian languages.
WORLD_ORDER = [
    "eng_Latn", "cmn_Hans", "hin_Deva", "arb_Arab", "fra_Latn", "ben_Beng", "por_Latn", "rus_Cyrl",
    "urd_Arab", "ind_Latn", "deu_Latn", "jpn_Jpan", "pcm_Latn", "mar_Deva", "tel_Telu", "tur_Latn",
    "tam_Taml", "yue_Hant", "vie_Latn", "tgl_Latn", "kor_Hang", "pes_Arab", "hau_Latn", "swh_Latn",
    "jav_Latn", "ita_Latn", "pnb_Arab", "guj_Gujr", "tha_Thai", "kan_Knda", "amh_Ethi", "pan_Guru",
    "pol_Latn", "yor_Latn", "ukr_Cyrl", "zsm_Latn", "mal_Mlym", "ory_Orya", "mya_Mymr", "npi_Deva",
    "sin_Sinh", "ron_Latn", "nld_Latn", "ibo_Latn", "ceb_Latn", "azj_Latn", "uzn_Latn", "khm_Khmr",
    "som_Latn", "zul_Latn", "ell_Grek", "hun_Latn", "ces_Latn", "swe_Latn", "heb_Hebr", "cat_Latn",
    "kin_Latn", "xho_Latn", "bel_Cyrl", "kaz_Cyrl", "srp_Cyrl", "hrv_Latn", "bul_Cyrl", "dan_Latn",
    "fin_Latn", "slk_Latn", "nob_Latn", "grn_Latn", "gug_Latn", "hat_Latn", "lin_Latn", "wol_Latn",
]

MACROAREA_ES = {
    "South America": "Sudamérica", "North America": "Norteamérica y Centroamérica", "Africa": "África",
    "Eurasia": "Europa y Asia", "Papunesia": "Oceanía y Sudeste Asiático", "Australia": "Australia",
}

WHISPER_STRONG_MAX_WER = 15.0   # FLEURS large-v3 WER/CER (%) at or below this = Whisper route


# ---------------------------------------------------------------------------
def fetch(key: str, binary: bool = False, timeout: float = 120.0):
    import requests
    CACHE.mkdir(parents=True, exist_ok=True)
    url = SOURCES[key]
    path = CACHE / (key + (".bin" if binary else ".txt"))
    if not path.exists() or path.stat().st_size == 0:
        print("  downloading", key, url, flush=True)
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path.read_bytes() if binary else path.read_text(encoding="utf-8")


def fetch_mms_tts() -> list:
    """All facebook/mms-tts-* repos (the API pages at 1000; follow the Link header)."""
    import requests
    path = CACHE / "mms_tts_models.json"
    if path.exists():
        return json.loads(path.read_text())
    url, ids = SOURCES["mms_tts_api"], []
    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        ids += [m["id"] for m in r.json()]
        nxt = r.links.get("next", {}).get("url")
        url = nxt
    ids = sorted(set(i for i in ids if re.fullmatch(r"facebook/mms-tts-[a-z]{3}", i)))
    path.write_text(json.dumps(ids))
    return ids


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def parse_whisper_languages(src: str) -> dict:
    block = src.split("LANGUAGES = {", 1)[1].split("}", 1)[0]
    return dict(re.findall(r'"([a-z]{2,3})":\s*"([^"]+)"', block))


def parse_whisper_fleurs(svg: str, whisper_langs: dict) -> dict:
    """FLEURS WER/CER per language for large-v3 from the SVG chart comments.

    The chart (matplotlib) lists 62 language labels sorted by large-v3 error,
    then the 62 large-v3 values in the same order. Returns {whisper_code: value}.
    """
    comments = re.findall(r"<!-- (.*?) -->", svg)
    idx = [i for i, c in enumerate(comments) if c.startswith("WER or")]
    start = idx[1] + 1                          # second panel = FLEURS
    end = comments.index("FLEURS", start)
    labels = [re.sub(r"[$]|~\(.*\)", "", c).strip() for c in comments[start:end]]
    after = comments.index("large-v2", end) + 1
    values = [float(c) for c in comments[after:after + len(labels)]]
    assert len(values) == len(labels), (len(values), len(labels))
    name_to_code = {v.lower(): k for k, v in whisper_langs.items()}
    name_to_code.update({"mandarin": "zh", "filipino": "tl", "norwegian": "no", "maori": "mi",
                         "punjabi": "pa", "persian": "fa", "cantonese": "yue"})
    out = {}
    for lab, val in zip(labels, values):
        code = name_to_code.get(lab.lower())
        if not code:
            raise SystemExit("unknown FLEURS label %r" % lab)
        out[code] = val
    return out


def main() -> int:
    t0 = time.time()
    print("fetching sources into", CACHE)
    # ISO 639-3
    iso = {}
    part1 = {}
    for row in csv.DictReader(io.StringIO(fetch("iso6393")), delimiter="\t"):
        iso[row["Id"]] = row
        if row["Part1"]:
            part1[row["Part1"]] = row["Id"]
    macro_members = {}
    for row in csv.DictReader(io.StringIO(fetch("iso6393_macro")), delimiter="\t"):
        if row["I_Status"] == "A":
            macro_members.setdefault(row["M_Id"], []).append(row["I_Id"])
    # Glottolog
    glotto = {}
    for row in csv.DictReader(io.StringIO(fetch("glottolog"))):
        if row["ISO639P3code"] and row["Level"] == "language":
            glotto[row["ISO639P3code"]] = row
    # Whisper
    whisper = parse_whisper_languages(fetch("whisper_tokenizer"))
    fleurs = parse_whisper_fleurs(fetch("whisper_breakdown"), whisper)
    # Omnilingual
    omni = re.findall(r'"([a-z]{3}_[A-Z][a-z]{3})"', fetch("omni_langs"))
    omni_cer = {}
    for row in csv.DictReader(io.StringIO(fetch("omni_cer"))):
        try:
            omni_cer[row["Language"]] = (float(row["CER"]), float(row["Training Hours"]))
        except (ValueError, KeyError):
            pass
    # MMS
    mms_tts = [i.rsplit("-", 1)[1] for i in fetch_mms_tts()]
    mms_lid = sorted(set(json.loads(fetch("mms_lid"))["id2label"].values()))
    # NLLB
    nllb_src = fetch("nllb_tokenizer")
    nllb_block = nllb_src.split("FAIRSEQ_LANGUAGE_CODES = [", 1)[1].split("]", 1)[0]
    nllb = re.findall(r"'([a-z]{3}_[A-Z][a-z]{3})'", nllb_block) or re.findall(r'"([a-z]{3}_[A-Z][a-z]{3})"', nllb_block)
    # MADLAD
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_proto=fetch("madlad_spm", binary=True))
    madlad = sorted({sp.id_to_piece(i)[2:-1] for i in range(sp.get_piece_size())
                     if re.fullmatch(r"<2[A-Za-z_\-]+>", sp.id_to_piece(i))})
    # Piper
    piper = json.loads(fetch("piper_voices"))
    print("sources: iso=%d glottolog=%d whisper=%d fleurs=%d omni=%d omni_cer=%d mms_tts=%d mms_lid=%d nllb=%d "
          "madlad=%d piper=%d" % (len(iso), len(glotto), len(whisper), len(fleurs), len(omni), len(omni_cer),
                                  len(mms_tts), len(mms_lid), len(nllb), len(madlad), len(piper)))
    assert len(whisper) >= 99 and len(omni) > 1500 and len(nllb) >= 200 and len(mms_tts) > 1000

    def iso3_of(tag: str) -> str:
        base = re.split(r"[_\-]", tag)[0].lower()
        if len(base) == 2:
            return part1.get(base, "")
        return base if base in iso else ""

    entries: dict[str, dict] = {}

    def entry(code: str) -> dict:
        if code not in entries:
            i3, script = code.split("_")
            entries[code] = {"code": code, "iso639_3": i3, "script": script, "engines": {}, "quality": {}}
        return entries[code]

    scripts_of: dict[str, list] = {}
    for c in list(omni) + list(nllb):
        i3, s = c.split("_")
        scripts_of.setdefault(i3, [])
        if s not in scripts_of[i3]:
            scripts_of[i3].append(s)

    def canon_for_iso3(i3: str, prefer_script: str = "") -> str:
        ss = scripts_of.get(i3, [])
        if prefer_script and prefer_script in ss:
            return i3 + "_" + prefer_script
        if ss:
            return i3 + "_" + ss[0]
        for s, members in DEFAULT_SCRIPT.items():
            if i3 in members:
                return i3 + "_" + s
        return i3 + "_Latn"

    # Omnilingual ASR
    for c in omni:
        e = entry(c)
        e["engines"]["omniasr"] = c
        if c in omni_cer:
            e["quality"]["omniasr_7b_cer"] = omni_cer[c][0]
            e["quality"]["omniasr_train_hours"] = omni_cer[c][1]
    # NLLB (zho_* is Mandarin in NLLB's data)
    for c in nllb:
        canon = {"zho_Hans": "cmn_Hans", "zho_Hant": "cmn_Hant"}.get(c, c)
        entry(canon)["engines"]["nllb"] = c
    # Whisper
    unmapped = []
    for w in whisper:
        canon = WHISPER_TO_CANON_OVERRIDE.get(w)
        if not canon:
            i3 = part1.get(w, w if w in iso else "")
            if not i3:
                unmapped.append(w)
                continue
            if i3 in macro_members:
                unmapped.append(w)
                continue
            canon = canon_for_iso3(i3)
        e = entry(canon)
        e["engines"]["whisper"] = w
        e["whisper_code"] = w
        if w in fleurs:
            e["quality"]["whisper_large_v3_fleurs_wer"] = fleurs[w]
    if unmapped:
        raise SystemExit("Whisper codes without canonical mapping: %s" % unmapped)
    # Mandarin traditional also decodes with Whisper "zh"
    if "cmn_Hant" in entries:
        entries["cmn_Hant"]["engines"]["whisper"] = "zh"
        entries["cmn_Hant"]["whisper_code"] = "zh"
    # MADLAD: tags are ISO 639-1/3, sometimes with script/region. Macrolanguage
    # tags (qu, ay, ...) serve every member language we already know.
    madlad_by_iso3: dict[str, str] = {}
    for tag in madlad:
        parts = re.split(r"[_\-]", tag)
        i3 = iso3_of(tag)
        if not i3:
            continue
        script = parts[1] if len(parts) > 1 and len(parts[1]) == 4 else ""
        if script:
            canon = i3 + "_" + script.capitalize()
            entry(canon)["engines"]["madlad"] = tag
            continue
        madlad_by_iso3.setdefault(i3, tag)
    for i3, tag in madlad_by_iso3.items():
        targets = [c for c in entries if c.startswith(i3 + "_")]
        if not targets and i3 not in macro_members:
            targets = [canon_for_iso3(i3)]
        for c in targets:
            entry(c)["engines"].setdefault("madlad", tag)
        for m in macro_members.get(i3, []):
            for c in [c for c in entries if c.startswith(m + "_")]:
                entries[c]["engines"].setdefault("madlad", tag)
                entries[c]["engines"]["madlad_macro"] = True
    # MMS-TTS (ISO 639-3, no script): attach to the entries of that iso3
    for i3 in mms_tts:
        targets = [c for c in entries if c.startswith(i3 + "_")] or [canon_for_iso3(i3)]
        for c in targets:
            entry(c)["engines"]["mms_tts"] = i3
    # MMS-LID labels (ISO 639-3)
    lid_set = set(mms_lid)
    for c, e in entries.items():
        if e["iso639_3"] in lid_set:
            e["engines"]["mms_lid"] = e["iso639_3"]
    # Piper: pick voices per language family (ISO 639-1 or -3)
    quality_rank = {"high": 3, "medium": 2, "low": 1, "x_low": 0}
    for key, v in piper.items():
        fam = v["language"]["family"]
        i3 = part1.get(fam, fam)
        if fam == "zh":
            i3 = "cmn"
        elif fam == "ar":
            i3 = "arb"
        elif fam == "fa":
            i3 = "pes"
        elif fam == "sw":
            i3 = "swh"
        elif fam == "ne":
            i3 = "npi"
        elif fam == "no":
            i3 = "nob"
        elif fam == "lv":
            i3 = "lvs"
        elif fam == "et":
            i3 = "ekk"
        elif fam == "sq":
            i3 = "als"
        elif fam == "ku":
            i3 = "kmr"
        elif fam == "ms":
            i3 = "zsm"
        targets = [c for c in entries if c.startswith(i3 + "_")]
        if fam == "zh":
            targets = ["cmn_Hans"]
        if not targets:
            targets = [canon_for_iso3(i3)]
        onnx = [p for p in v["files"] if p.endswith(".onnx")]
        size = sum(f["size_bytes"] for f in v["files"].values())
        for c in targets:
            voices = entry(c)["engines"].setdefault("piper", [])
            voices.append({"key": key, "quality": v["quality"], "region": v["language"].get("region", ""),
                           "speakers": v.get("num_speakers", 1), "onnx": onnx[0] if onnx else "",
                           "size_mb": round(size / 1e6, 1)})
    for e in entries.values():
        if "piper" in e["engines"]:
            e["engines"]["piper"].sort(key=lambda x: (-quality_rank.get(x["quality"], 0), x["key"]))

    # Peru: make sure every listed language exists (even with no engine yet)
    peru = {}
    for rank, (i3, es, native, aliases) in enumerate(PERU, start=1):
        if i3 not in iso:
            raise SystemExit("Peru list: %s is not an ISO 639-3 code" % i3)
        canon = canon_for_iso3(i3)
        existing = [c for c in entries if c.startswith(i3 + "_")]
        if existing:
            canon = existing[0] if canon not in existing else canon
        e = entry(canon)
        peru[canon] = rank
        e["name_es"], e["name_native"] = es, native
        e["aliases"] = sorted(set(aliases))
        e["peru"] = True

    # Names, region, capabilities
    from babel import Locale
    loc_es, loc_en = Locale("es"), Locale("en")
    whisper_strong = {w for w, v in fleurs.items() if v <= WHISPER_STRONG_MAX_WER}
    for code, e in entries.items():
        i3 = e["iso639_3"]
        row = iso.get(i3)
        p1 = row["Part1"] if row else ""
        keys = [k for k in (e.get("whisper_code") or "", p1, i3) if k]
        name_en = next((loc_en.languages[k] for k in keys if k in loc_en.languages), "") or (row["Ref_Name"] if row else i3)
        e["name_en"] = e.get("name_en") or name_en
        if not e.get("name_es"):
            e["name_es"] = next((loc_es.languages[k] for k in keys if k in loc_es.languages), "") or e["name_en"]
            e["name_es"] = e["name_es"][:1].upper() + e["name_es"][1:]
        if not e.get("name_native"):
            native = ""
            for k in keys:
                try:
                    native = Locale.parse(k).languages.get(k, "")
                except Exception:
                    native = ""
                if native:
                    break
            e["name_native"] = (native[:1].upper() + native[1:]) if native else e["name_en"]
        if code in ("cmn_Hans", "cmn_Hant"):
            e["name_es"] = "Chino mandarín (%s)" % ("simplificado" if code.endswith("Hans") else "tradicional")
            e["name_en"] = "Mandarin Chinese (%s)" % ("Simplified" if code.endswith("Hans") else "Traditional")
            e["name_native"] = "普通话" if code.endswith("Hans") else "國語"
        g = glotto.get(i3)
        countries = sorted(set((g["Countries"] or "").split(";"))) if g else []
        countries = [c for c in countries if c]
        e["countries"] = countries
        areas = [a for a in ((g["Macroarea"] or "").split(";") if g else []) if a]
        e["region"] = "Perú" if e.get("peru") else ", ".join(MACROAREA_ES.get(a, a) for a in areas)
        e["iso639_1"] = p1
        eng = e["engines"]
        asr = []
        if "whisper" in eng:
            asr.append("whisper")
        if "omniasr" in eng:
            asr.append("omniasr")
        e["asr"] = asr
        e["whisper_strong"] = eng.get("whisper") in whisper_strong
        tts = []
        if eng.get("piper"):
            tts.append("piper")
        if "mms_tts" in eng:
            tts.append("mms")
        e["tts"] = tts
        e["translate"] = bool(eng.get("nllb") or eng.get("madlad"))
        e.setdefault("whisper_code", "")
        e.setdefault("aliases", [])
        e.setdefault("peru", False)

    # popular ranking: Spanish + Peru (curated order), then the world
    order = [c for c, _ in sorted(peru.items(), key=lambda kv: kv[1])]
    order += [c for c in WORLD_ORDER if c in entries and c not in order]
    for e in entries.values():
        e["popular"] = 0
    for i, c in enumerate(order, start=1):
        entries[c]["popular"] = i

    # Drop entries nobody can use (e.g. a script variant only a MADLAD tag knows)
    # unless they are Peruvian.
    keep = [e for e in entries.values() if e["asr"] or e["tts"] or e["translate"] or e["peru"]]
    field_order = ["code", "iso639_3", "iso639_1", "script", "name_native", "name_es", "name_en", "region",
                   "countries", "asr", "tts", "translate", "whisper_code", "whisper_strong", "popular", "peru",
                   "aliases", "engines", "quality"]
    keep = [{k: e.get(k) for k in field_order} for e in keep]
    keep.sort(key=lambda e: (e["popular"] == 0, e["popular"], strip_accents(e["name_es"])))

    codes = [e["code"] for e in keep]
    assert len(codes) == len(set(codes))
    meta = {
        "built": time.strftime("%Y-%m-%d"),
        "sources": SOURCES,
        "whisper_strong_rule": "Whisper large-v3 FLEURS WER/CER <= %.0f%% (openai/whisper language-breakdown.svg)"
                               % WHISPER_STRONG_MAX_WER,
        "counts": {
            "languages": len(keep),
            "asr_any": sum(1 for e in keep if e["asr"]),
            "asr_whisper": sum(1 for e in keep if "whisper" in e["asr"]),
            "asr_omniasr": sum(1 for e in keep if "omniasr" in e["asr"]),
            "tts_any": sum(1 for e in keep if e["tts"]),
            "tts_piper": sum(1 for e in keep if "piper" in e["tts"]),
            "tts_mms": sum(1 for e in keep if "mms" in e["tts"]),
            "translate": sum(1 for e in keep if e["translate"]),
            "translate_nllb": sum(1 for e in keep if e["engines"].get("nllb")),
            "translate_madlad": sum(1 for e in keep if e["engines"].get("madlad")),
            "peru": sum(1 for e in keep if e["peru"]),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "languages": keep}, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(json.dumps(meta["counts"], indent=1))
    print("wrote %s (%.0f KB) in %.0f s" % (OUT, OUT.stat().st_size / 1024, time.time() - t0))
    for e in keep[:12]:
        print("  %3d %-9s %-40s asr=%s tts=%s tr=%s" % (e["popular"], e["code"], e["name_es"], e["asr"], e["tts"],
                                                       e["translate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
