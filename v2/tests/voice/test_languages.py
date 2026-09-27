"""Language registry integrity (yq.common.languages)."""
import re

from yq.common import languages as L

PERU_CORE = ["spa_Latn", "quy_Latn", "quz_Latn", "qxp_Latn", "ayr_Latn", "cni_Latn", "agr_Latn", "shp_Latn",
             "mcb_Latn"]


def test_codes_unique_and_canonical():
    codes = [l.code for l in L.all()]
    assert len(codes) == len(set(codes))
    assert len(codes) > 1500
    for l in L.all():
        assert re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", l.code), l.code
        assert l.code.split("_")[0] == l.iso639_3
        assert l.name_es and l.name_en and l.name_native
        assert set(l.asr) <= {"whisper", "omniasr"} and set(l.tts) <= {"piper", "mms"}


def test_every_engine_code_maps_back():
    seen = {"whisper": set(), "omniasr": set(), "nllb": set()}
    for l in L.all():
        for eng in seen:
            code = l.engines.get(eng)
            if code:
                assert code not in seen[eng] or eng == "whisper", (eng, code)   # zh serves Hans + Hant
                seen[eng].add(code)
                assert L.get(code) is not None, (eng, code)
        if l.whisper_code:
            assert "whisper" in l.asr and L.get(l.whisper_code) is not None
        if "omniasr" in l.asr:
            assert L.get(l.engines["omniasr"]).code == l.code
        if "mms" in l.tts:
            assert re.fullmatch(r"[a-z]{3}", l.engines["mms_tts"])
        if "piper" in l.tts:
            assert l.engines["piper"] and l.engines["piper"][0]["onnx"].endswith(".onnx")
        if l.translate:
            assert l.engines.get("nllb") or l.engines.get("madlad")
    assert len(seen["whisper"]) >= 99 and len(seen["omniasr"]) > 1600 and len(seen["nllb"]) >= 200


def test_peruvian_languages_present_and_first():
    for c in PERU_CORE:
        l = L.get(c)
        assert l is not None and l.peru and l.region == "Perú", c
        assert "omniasr" in l.asr, c
    top = [l.code for l in L.all()[:5]]
    assert top[0] == "spa_Latn" and top[1:3] == ["quy_Latn", "quz_Latn"]
    ranks = [l.popular for l in L.all() if l.popular]
    assert ranks == sorted(ranks) and ranks[0] == 1
    peru_max = max(l.popular for l in L.peru())
    assert L.get("eng_Latn").popular > peru_max          # Peru first, then the world
    assert L.get("quy_Latn").translate and L.get("ayr_Latn").translate
    assert "mms" in L.get("quy_Latn").tts and "mms" in L.get("ayr_Latn").tts


def test_whisper_facts():
    es, en = L.get("es"), L.get("en")
    assert es.code == "spa_Latn" and es.whisper_strong and "whisper" in es.asr
    assert en.code == "eng_Latn" and en.whisper_code == "en"
    assert L.get("quy_Latn").whisper_code == "" and not L.get("quy_Latn").whisper_strong
    assert L.get("zh").code == "cmn_Hans" and L.get("yue").code == "yue_Hant"


def test_code_resolution_variants():
    assert L.resolve("es-PE") == "spa_Latn"
    assert L.resolve("SPA") == "spa_Latn"
    assert L.resolve("quy") == "quy_Latn"
    assert L.resolve("qu") == "quy_Latn" and L.resolve("ay") == "ayr_Latn"
    assert L.resolve("zho_Hans") == "cmn_Hans"            # NLLB's code for Mandarin
    assert L.get("xx_nope") is None and L.resolve("") is None


def test_search_with_and_without_accents():
    assert L.search("aimara")[0].code == "ayr_Latn"
    assert L.search("Aymara")[0].code == "ayr_Latn"
    assert L.search("quechua cusco")[0].code == "quz_Latn"
    assert L.search("QUECHUA CUSQUEÑO")[0].code == "quz_Latn"
    assert L.search("ashaninka")[0].code == "cni_Latn"
    assert L.search("asháninka")[0].code == "cni_Latn"
    assert L.search("japones")[0].code == "jpn_Jpan" and L.search("japonés")[0].code == "jpn_Jpan"
    assert L.search("日本語")[0].code == "jpn_Jpan"
    assert L.search("English")[0].code == "eng_Latn"
    assert L.search("runasimi")[0].peru
    assert L.search("zzzzqqq") == []


def test_ui_list_features():
    asr = L.ui_list("asr")
    assert asr[0]["code"] == "spa_Latn" and all(d["asr"] for d in asr)
    tts = L.ui_list("tts")
    assert all(d["tts"] for d in tts) and any(d["code"] == "quy_Latn" for d in tts)
    tr = L.ui_list("translate")
    assert all(d["translate"] for d in tr)
    assert {"code", "name", "name_es", "name_native", "name_en", "popular", "asr", "tts", "translate"} <= set(asr[0])
    try:
        L.ui_list("nope")
        assert False
    except ValueError:
        pass


def test_story_feature_needs_hearing_and_understanding():
    """A story language must be recognized AND translatable, or the drawing can't follow the story."""
    story = {d["code"] for d in L.ui_list("story")}
    asr = {d["code"] for d in L.ui_list("asr")}
    tr = {d["code"] for d in L.ui_list("translate")}
    assert story == asr & tr
    assert {"spa_Latn", "eng_Latn", "quy_Latn", "quz_Latn", "ayr_Latn", "agr_Latn", "shp_Latn"} <= story
    assert asr - story, "some languages can only be written down, not understood"
    assert all(L.get(c).has("asr") and L.get(c).has("translate") for c in story)
