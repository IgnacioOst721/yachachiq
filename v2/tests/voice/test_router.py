"""ASR routing decisions per language and availability (no models)."""
from yq.macworker.models import voice_router as vr

W, O = "whisper-large-v3", "omniasr-ctc-1b"


def test_whisper_strong_language_goes_to_whisper_first():
    p = vr.plan_for_language("spa_Latn", W, O)
    assert p.lang == "spa_Latn" and p.engines == [W, O]
    p = vr.plan_for_language("en", W, O)
    assert p.lang == "eng_Latn" and p.engines[0] == W


def test_quechua_aymara_amazonian_go_to_omnilingual():
    for code in ("quy_Latn", "quz_Latn", "qxp_Latn", "ayr_Latn", "cni_Latn", "shp_Latn", "agr_Latn", "mcb_Latn"):
        p = vr.plan_for_language(code, W, O)
        assert p.engines == [O], (code, p)


def test_weak_whisper_language_prefers_omnilingual_with_whisper_backup():
    p = vr.plan_for_language("swh_Latn", W, O)          # Whisper FLEURS WER ~34 %: weak
    assert p.engines == [O, W]


def test_availability_changes_the_plan():
    assert vr.plan_for_language("spa_Latn", W, O, available={O}).engines == [O]
    assert vr.plan_for_language("quy_Latn", W, O, available={W}).engines == []
    assert vr.plan_for_language("spa_Latn", W, O, available={W}).engines == [W]


def test_auto_prefers_mms_lid_for_quechua_even_if_whisper_says_spanish():
    p = vr.plan_auto([["spa_Latn", 0.8], ["por_Latn", 0.1]], [["quy_Latn", 0.7], ["spa_Latn", 0.2]], W, O)
    assert p.lang == "quy_Latn" and p.engines == [O]
    assert p.candidates[0][0] in ("spa_Latn", "quy_Latn") and len(p.candidates) >= 2


def test_auto_whisper_confident_spanish():
    p = vr.plan_auto([["spa_Latn", 0.97]], [["spa_Latn", 0.9]], W, O)
    assert p.lang == "spa_Latn" and p.engines[0] == W


def test_auto_unsure_uses_language_free_ctc():
    p = vr.plan_auto([["spa_Latn", 0.3]], [["quy_Latn", 0.2], ["quz_Latn", 0.2]], W, O)
    assert p.engines[0] == O and p.reason == "lid unsure"
    p = vr.plan_auto([], [], W, "omniasr-llm-1b")
    assert p.engines[0] == W


def test_merge_lid_ignores_whisper_for_weak_languages():
    c = vr.merge_lid([["swh_Latn", 0.9], ["eng_Latn", 0.05]], [["quy_Latn", 0.6]])
    codes = [x[0] for x in c]
    assert "swh_Latn" not in codes and codes[0] == "quy_Latn"


def test_auto_sums_quechua_varieties_before_deciding():
    # MMS-LID hesitates between Quechua varieties; together they are clearly Quechua
    p = vr.plan_auto([["spa_Latn", 0.7]], [["quz_Latn", 0.3], ["quy_Latn", 0.25], ["spa_Latn", 0.2]], W, O)
    assert p.lang == "quz_Latn" and p.engines == [O]
    assert vr.lang_group("qxp_Latn") == vr.lang_group("quy_Latn") == "quechua"
    assert vr.lang_group("ayr_Latn") == "aymara" and vr.lang_group("spa_Latn") == "spa_Latn"
