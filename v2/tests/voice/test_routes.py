"""Mac worker voice endpoints with fake models (no weights loaded)."""
import io
import wave

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yq.macworker import routes_voice as rv
from yq.macworker.modelmgr import ModelManager
from yq.voice import audio as au
from yq.voice import settings
from yq.voice.capture import synthetic_speech

WHISPER = settings.MAC_WHISPER_MODEL


class FakeWhisper:
    def __init__(self, lid):
        self.lid, self.calls = lid, []

    def detect_language(self, x):
        return self.lid

    def transcribe(self, x, language=None, prompt=""):
        self.calls.append(language)
        return {"text": "hola mundo", "language": language or self.lid[0][0], "confidence": 0.9,
                "segments": [{"start": 0.0, "end": 1.0, "text": "hola mundo", "confidence": 0.9}]}


class FakeOmni:
    def __init__(self):
        self.calls = []

    def transcribe(self, x, lang=None):
        self.calls.append(lang)
        return {"text": "ñuqaqa runam kani", "confidence": 0.8}


class FakeLID:
    def __init__(self, out):
        self.out = out

    def predict(self, x, allowed_iso3=None, top=5):
        return [o for o in self.out if not allowed_iso3 or o[0] in allowed_iso3]


class FakeMT:
    def translate_batch(self, texts, s, t, beam=4):
        return ["[%s>%s] %s" % (s, t, x) for x in texts]


class FakeTTS:
    def synthesize(self, text):
        return np.zeros(1600, np.float32), 16000


@pytest.fixture
def client(monkeypatch):
    mm = ModelManager(budget_gb=100)
    fakes = {"whisper": FakeWhisper([["es", 0.95], ["pt", 0.03]]), "omni": FakeOmni(),
             "lid": FakeLID([["spa", 0.8], ["quy", 0.1]])}
    mm.register(WHISPER, loader=lambda: fakes["whisper"], size_gb=1)
    mm.register("omniasr-ctc-1b", loader=lambda: fakes["omni"], size_gb=1)
    mm.register(rv.LID_MODEL, loader=lambda: fakes["lid"], size_gb=1)
    mm.register("mt-nllb", loader=FakeMT, size_gb=1)
    mm.register("mt-madlad", loader=FakeMT, size_gb=1)
    mm.register("mms-tts-quy", loader=FakeTTS, size_gb=0.1)
    mm.register("mms-tts-spa", loader=FakeTTS, size_gb=0.1)
    monkeypatch.setattr(rv, "MODELS", mm)
    app = FastAPI()
    app.include_router(rv.router)
    c = TestClient(app)
    c.fakes = fakes
    return c


def wav(x=None):
    return {"audio": ("a.wav", au.to_wav_bytes(synthetic_speech(1.5) if x is None else x), "audio/wav")}


def test_asr_spanish_uses_whisper(client):
    r = client.post("/asr", files=wav(), data={"lang": "spa_Latn"}).json()
    assert r["engine"] == WHISPER + "@mac" and r["lang"] == "spa_Latn" and r["text"] == "hola mundo"
    assert client.fakes["whisper"].calls == ["es"]
    assert 0 <= r["confidence"] <= 1 and r["duration_s"] == 1.5


def test_asr_quechua_uses_omnilingual_with_its_code(client):
    r = client.post("/asr", files=wav(), data={"lang": "quy"}).json()
    assert r["engine"] == "omniasr-ctc-1b@mac" and r["lang"] == "quy_Latn"
    assert client.fakes["omni"].calls == ["quy_Latn"]


def test_asr_auto_quechua_detected_by_mms_lid(client):
    client.fakes["lid"].out = [["quy", 0.75], ["spa", 0.2]]
    r = client.post("/asr", files=wav(), data={"lang": "auto"}).json()
    assert r["lang"] == "quy_Latn" and r["engine"].startswith("omniasr")
    assert r["lang_candidates"][0][0] in ("quy_Latn", "spa_Latn")


def test_asr_auto_spanish(client):
    r = client.post("/asr", files=wav(), data={"lang": "auto"}).json()
    assert r["lang"] == "spa_Latn" and r["engine"].startswith("whisper")


def test_asr_rejects_bad_input(client):
    assert client.post("/asr", files={"audio": ("a.wav", b"not a wav", "audio/wav")}).status_code == 400
    assert client.post("/asr", files=wav(), data={"lang": "xx_Nope"}).status_code == 400


def test_asr_falls_back_when_first_engine_breaks(client):
    def boom(*a, **k):
        raise RuntimeError("gpu on fire")
    client.fakes["whisper"].transcribe = boom
    r = client.post("/asr", files=wav(), data={"lang": "spa_Latn"}).json()
    assert r["engine"] == "omniasr-ctc-1b@mac"


def test_lid_endpoint(client):
    r = client.post("/lid", files=wav()).json()
    assert r["candidates"][0][0] == "spa_Latn" and 0 < r["candidates"][0][1] <= 1


def test_translate_endpoint_picks_engine(client):
    r = client.post("/translate", json={"text": "Hola. Adiós.", "src": "spa_Latn", "tgt": "quy_Latn"}).json()
    assert r["engine"] == "nllb" and r["text"].startswith("[spa_Latn>quy_Latn]")
    r = client.post("/translate", json={"text": "hola", "src": "es", "tgt": "es"}).json()
    assert r == {"text": "hola", "engine": "same"}
    assert client.post("/translate", json={"text": "x", "src": "spa_Latn", "tgt": "jqr_Latn"}).status_code == 400


def test_tts_endpoint_returns_wav(client):
    r = client.post("/tts", json={"text": "Allinllachu", "lang": "quy_Latn"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 1600
    assert r.headers["x-tts-engine"] == "mms-tts-quy"
    assert client.post("/tts", json={"text": "", "lang": "quy_Latn"}).status_code == 400
    assert client.post("/tts", json={"text": "x", "lang": "xx_Nope"}).status_code == 404


def test_voice_domain_loads_in_worker_app():
    from yq.macworker.app import DOMAINS, create_app
    c = TestClient(create_app())
    h = c.get("/health").json()
    assert h["domains"].get("routes_voice") == "ok", h["domains"]
    models = h["models"]["models"]
    assert WHISPER in models and "whisper-large-v3" in models and "omniasr-ctc-1b" in models and "mt-nllb" in models
    assert not any(m["loaded"] for m in models.values())      # nothing heavy loaded at startup
