"""Heavy smoke test: macOS `say` speaks Spanish and English, the real Mac models transcribe it.

Run inside the memory lock:
    .venvs/voice/bin/python -m yq.common.heavylock .venvs/voice/bin/python -m pytest -m heavy tests/voice
"""
import shutil
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.heavy, pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"),
                                                    reason="needs macOS say")]

SENTENCES = {
    "spa_Latn": (["Paulina", "Mónica", "Eddy (Spanish (Mexico))"],
                 "Había una vez una niña que vivía en las montañas con su llama y un cóndor."),
    "eng_Latn": (["Samantha", "Alex", "Eddy (English (US))"],
                 "Once upon a time a girl lived in the mountains with her llama and a condor."),
}


def speak(text, voices, path):
    for v in voices:
        r = subprocess.run(["say", "-v", v, "-o", str(path), "--data-format=LEI16@16000", text],
                           capture_output=True)
        if r.returncode == 0 and path.exists() and path.stat().st_size > 1000:
            return path
    pytest.skip("no macOS voice among %s" % voices)


@pytest.fixture(scope="module")
def mac_models():
    from yq.macworker import routes_voice as rv
    from yq.macworker.modelmgr import ModelManager
    mm = ModelManager(budget_gb=9)
    rv.register_models(mm)
    old = rv.MODELS
    rv.MODELS = mm
    yield rv
    mm.unload_all()
    rv.MODELS = old


@pytest.mark.parametrize("lang", ["spa_Latn", "eng_Latn"])
@pytest.mark.parametrize("mode", ["given", "auto"])
def test_say_then_transcribe(tmp_path, mac_models, lang, mode):
    import jiwer
    from yq.voice import audio as au
    from yq.voice.textnorm import normalize
    voices, text = SENTENCES[lang]
    x = au.load_audio(speak(text, voices, tmp_path / "say.wav"))
    t = mac_models.transcribe(x, lang if mode == "given" else "auto")
    wer = jiwer.wer(normalize(text), normalize(t.text))
    print(lang, mode, t.engine, t.lang, "%.2f" % t.confidence, "WER %.2f" % wer, "|", t.text)
    assert t.lang == lang
    assert wer <= 0.2, t.text
    assert 0.3 <= t.confidence <= 1.0


def test_omnilingual_hears_spanish(tmp_path, mac_models):
    import jiwer
    from yq.voice import audio as au
    from yq.voice.textnorm import normalize
    from yq.voice import settings
    voices, text = SENTENCES["spa_Latn"]
    x = au.load_audio(speak(text, voices, tmp_path / "say.wav"))
    t = mac_models.run_engine(settings.MAC_OMNI_MODEL, x, "spa_Latn", "")
    cer = jiwer.cer(normalize(text), normalize(t.text))
    print("omni", t.engine, "CER %.2f" % cer, "|", t.text)
    assert cer <= 0.15, t.text


def test_translate_and_tts_real(mac_models):
    from yq.macworker.models import voice_translate as vt
    out = vt.translate_detail("El cóndor vuela sobre la montaña.", "spa_Latn", "eng_Latn",
                              get_model=lambda e: mac_models.MODELS.get("mt-" + e))
    assert "condor" in out["text"].lower() and out["engine"] == "nllb"
    x, rate, eng = mac_models.synthesize("Allinllachu kachkanki", "quy_Latn")
    assert eng == "mms-tts-quy" and rate == 16000 and len(x) > rate * 0.5
