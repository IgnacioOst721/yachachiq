"""Small pure helpers: translation routing/splitting, text normalisation, Omnilingual chunking."""
import hashlib
import sys
from pathlib import Path

import numpy as np

from yq.macworker.models import voice_omni, voice_translate as vt
from yq.voice.textnorm import normalize


def test_translation_engine_choice():
    assert vt.pick_engine("spa_Latn", "quy_Latn") == "nllb"
    assert vt.pick_engine("ayr_Latn", "spa_Latn") == "nllb"
    assert vt.engine_codes("cmn_Hans", "spa_Latn", "nllb") == ("zho_Hans", "spa_Latn")
    # Awajún / Shipibo: only MADLAD has them
    assert vt.pick_engine("agr_Latn", "spa_Latn") == "madlad"
    assert vt.engine_codes("shp_Latn", "spa_Latn", "madlad") == ("shp", "es")
    # Asháninka: nobody translates it (honest: no engine)
    assert vt.pick_engine("cni_Latn", "spa_Latn") is None
    assert vt.pick_engine("spa_Latn", "quy_Latn", available=["madlad"]) == "madlad"


def test_translate_detail_with_fake_model():
    class Fake:
        def translate_batch(self, texts, s, t, beam=4):
            return [x.upper() for x in texts]
    out = vt.translate_detail("hola. chau.", "es", "quy", get_model=lambda e: Fake())
    assert (out["text"], out["engine"], out["verified"]) == ("HOLA. CHAU.", "nllb", True)   # + additive keys
    assert vt.translate_detail("x", "es", "spa_Latn", get_model=None)["engine"] == "same"
    try:
        vt.translate_detail("x", "spa_Latn", "cni_Latn", get_model=lambda e: Fake())
        assert False
    except ValueError:
        pass


def test_split_sentences_keeps_everything():
    text = "Había una vez un cóndor. " * 40 + "Fin sin punto " + "palabra " * 120
    parts = vt.split_sentences(text, max_chars=200)
    assert all(len(p) <= 200 for p in parts)
    assert " ".join(parts).split() == text.split()
    assert vt.split_sentences("") == [] and vt.split_sentences("hola") == ["hola"]


def test_normalize_for_wer():
    assert normalize("¡Hola, T'ika!  Ch’aska 'dijo'…") == "hola t'ika ch'aska dijo"
    assert normalize("ÑUQAQA Runam Kani.") == "ñuqaqa runam kani"
    assert normalize("") == ""


def test_omni_split_long_audio_under_40s():
    sr = 16000
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(100 * sr)).astype(np.float32)
    for k in range(5, 100, 7):                  # quiet gaps every 7 s
        x[k * sr: k * sr + sr // 5] *= 0.001
    chunks = voice_omni.split_long(x)
    assert all(len(c) <= 38 * sr for c in chunks) and sum(len(c) for c in chunks) == len(x)
    assert len(voice_omni.split_long(x[: 10 * sr])) == 1


def test_fairseq2_cache_path_matches_library_rule():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import voice_download
    url = "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B-v2.pt"
    p = voice_download.fairseq2_asset_path(url)
    assert p.parent.name == hashlib.sha1(url.encode()).hexdigest()[:24] and p.name == "omniASR-CTC-1B-v2.pt"
