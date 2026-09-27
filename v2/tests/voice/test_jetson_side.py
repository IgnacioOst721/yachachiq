"""Jetson-side voice modules: ASR chain, TTS fallback chain, translation fallback, recorder."""
import threading
import time

import numpy as np
import pytest

from yq.common import config
from yq.common.macclient import MacUnavailable
from yq.voice import asr, audio as au, capture, settings, translate, tts
from yq.voice.capture import synthetic_speech


# ---------------------------------------------------------------- ASR
class FakeClient:
    def __init__(self, fail=False, payload=None):
        self.fail, self.payload, self.calls = fail, payload, []

    def post_files(self, path, files, data=None, timeout=None):
        self.calls.append((path, data, timeout))
        if self.fail:
            raise MacUnavailable("down")
        return self.payload

    def post_json(self, path, payload, timeout=None):
        self.calls.append((path, payload, timeout))
        if self.fail:
            raise MacUnavailable("down")
        return {"text": "Un cóndor", "engine": "nllb"}


def test_asr_uses_mac_first(monkeypatch):
    fc = FakeClient(payload={"text": "hola", "lang": "spa_Latn", "engine": "whisper-large-v3@mac",
                             "confidence": 0.9, "duration_s": 1.5, "future_field": 1})
    monkeypatch.setattr("yq.common.macclient.client", lambda: fc)
    t = asr.transcribe(synthetic_speech(1.5), lang="es")
    assert t.text == "hola" and t.engine.endswith("@mac")
    assert fc.calls[0][0] == "/asr" and fc.calls[0][1]["lang"] == "spa_Latn"


def test_asr_falls_back_to_local_then_none(monkeypatch):
    monkeypatch.setattr("yq.common.macclient.client", lambda: FakeClient(fail=True))
    tried = []

    def fake_local(x, lang, prompt, kind):
        tried.append(kind)
        raise RuntimeError("no faster-whisper here")
    monkeypatch.setattr(asr, "_faster_whisper", fake_local)
    monkeypatch.setattr(asr, "cuda_available", lambda: True)
    t = asr.transcribe(synthetic_speech(1.0), lang="auto")
    assert tried == ["gpu", "cpu"] and t.engine == "none" and t.text == "" and t.confidence == 0.0


def test_asr_local_timeout_never_hangs(monkeypatch):
    monkeypatch.setattr(config, "MOCK", False)
    monkeypatch.setenv("YQ_MOCK_MAC", "1")
    monkeypatch.setattr(settings, "ASR_LOCAL_TIMEOUT_S", 0.3)
    monkeypatch.setattr(asr, "cuda_available", lambda: False)
    monkeypatch.setattr(asr, "_faster_whisper", lambda *a: time.sleep(5))
    t0 = time.time()
    t = asr.transcribe(synthetic_speech(1.0))
    assert time.time() - t0 < 2.0 and t.engine == "none"


def test_asr_mock_mode(monkeypatch):
    monkeypatch.setenv("YQ_MOCK_ASR", "1")
    t = asr.transcribe(np.zeros(16000, np.float32))
    assert t.engine == "mock" and "cóndor" in t.text and t.lang == "spa_Latn"


def test_asr_empty_audio_is_not_sent(monkeypatch):
    fc = FakeClient(payload={})
    monkeypatch.setattr("yq.common.macclient.client", lambda: fc)
    t = asr.transcribe(np.zeros(100, np.float32))
    assert t.engine == "none" and fc.calls == []


def test_segments_confidence_bounds():
    assert asr.segments_confidence([]) == 0.0
    c = asr.segments_confidence([{"start": 0, "end": 1, "avg_logprob": -0.1, "no_speech_prob": 0.0},
                                 {"start": 1, "end": 3, "avg_logprob": -2.0, "no_speech_prob": 0.5}])
    assert 0 < c < 1


# ---------------------------------------------------------------- translation
def test_translate_mac_and_fallback(monkeypatch):
    monkeypatch.setattr("yq.common.macclient.client", lambda: FakeClient())
    assert translate.translate("Un cóndor", "es", "quy") == "Un cóndor"
    assert translate.translate_detail("x", "spa_Latn", "quy_Latn")["engine"] == "nllb"
    monkeypatch.setattr("yq.common.macclient.client", lambda: FakeClient(fail=True))
    assert translate.translate_detail("Allin", "quy_Latn", "spa_Latn") == {"text": "Allin", "engine": "none"}
    assert translate.translate_detail("hola", "es", "spa_Latn")["engine"] == "same"


# ---------------------------------------------------------------- TTS
def test_tts_chain_piper_then_mac_then_silent(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PIPER_DIR", str(tmp_path))
    monkeypatch.setenv("YQ_MOCK_SPEAKER", "1")
    # 1. no piper voice, Mac mocked away -> False, silent
    monkeypatch.setenv("YQ_MOCK_MAC", "1")
    assert tts.say("Allinllachu", "quy_Latn") is False and tts.last_engine == "none"
    # 2. Mac answers -> spoken through the Mac
    monkeypatch.delenv("YQ_MOCK_MAC")
    monkeypatch.setattr(tts, "synthesize_mac", lambda text, lang: (np.zeros(800, np.float32), 16000))
    assert tts.say("Allinllachu", "quy_Latn", wait=True) is True and tts.last_engine == "mac"
    # 3. a local Piper voice exists -> used first
    key = tts.candidate_voices("spa_Latn")[0]
    for p in tts.voice_files(key):
        p.write_text("{}")
    monkeypatch.setattr(tts, "synthesize_piper", lambda text, k: (np.zeros(800, np.float32), 22050))
    assert tts.say("Hola", "spa_Latn", wait=True) is True and tts.last_engine == "piper:" + key
    assert tts.say("", "spa_Latn") is False


def test_tts_mutes_microphone_while_speaking(monkeypatch):
    monkeypatch.setenv("YQ_MOCK_SPEAKER", "1")
    monkeypatch.setattr(tts, "synthesize", lambda text, lang: (np.zeros(8000, np.float32), 16000, "fake"))
    assert tts.say("hola", "spa_Latn") is True
    time.sleep(0.1)
    assert au.is_speaking()
    tts.stop()
    time.sleep(au.SPEAK_GUARD_S + 0.1)
    assert not au.is_speaking()


def test_best_spanish_voice_is_high_quality():
    keys = tts.candidate_voices("spa_Latn")
    assert keys[0] == "es_MX-claude-high"


# ---------------------------------------------------------------- recorder with a fake sound card
class FakeStream:
    def __init__(self, signal, samplerate, blocksize, callback, **kw):
        self.signal, self.rate, self.block, self.cb = signal, samplerate, blocksize, callback
        self._stop = threading.Event()

    def start(self):
        def run():
            for i in range(0, len(self.signal), self.block):
                if self._stop.is_set():
                    return
                chunk = self.signal[i:i + self.block]
                if len(chunk) < self.block:
                    chunk = np.pad(chunk, (0, self.block - len(chunk)))
                self.cb(chunk.reshape(-1, 1), len(chunk), None, None)
                time.sleep(self.block / self.rate / 20)          # 20x real time
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def close(self):
        pass


class FakeSD:
    def __init__(self, signal, rate=48000):
        self.signal, self.rate = signal, rate

    def query_devices(self, dev=None, kind=None):
        d = {"name": "SPEAKPHONE SP300U: USB Audio (hw:2,0)", "max_input_channels": 1, "max_output_channels": 2,
             "default_samplerate": float(self.rate)}
        return d if dev is not None or kind else [{"name": "HDA Intel", "max_input_channels": 2,
                                                   "max_output_channels": 2, "default_samplerate": 44100.0}, d]

    def check_input_settings(self, **kw):
        if kw.get("samplerate") != self.rate:
            raise ValueError("unsupported rate")

    def InputStream(self, samplerate, channels, dtype, device, blocksize, callback):
        return FakeStream(self.signal, samplerate, blocksize, callback)


def test_recorder_finds_speakerphone_resamples_and_endpoints(monkeypatch):
    sil = np.zeros(int(1.0 * 48000), np.float32)
    speech48 = au.resample(synthetic_speech(2.0), 16000, 48000)
    sd = FakeSD(np.concatenate([sil, speech48, sil, sil, sil]))
    monkeypatch.setattr(au, "_sd", lambda: sd)
    monkeypatch.setattr(settings, "VAD_SILENCE_S", 1.0)
    rec = capture.Recorder()
    levels = []
    x = rec.record(on_level=levels.append, max_s=20)
    try:
        import onnxruntime  # noqa: F401
        expected_vad = "silero"
    except ImportError:
        expected_vad = "energy"
    assert rec.last_device.startswith("SPEAKPHONE") and rec.last_reason == "silence" and rec.last_vad == expected_vad
    assert 2.0 < len(x) / 16000 < 3.5 and max(levels) > 0.5
    assert au.find_device("input", "", settings.MIC_NAMES) == 1


def test_recorder_stop_event_and_mock(monkeypatch):
    sd = FakeSD(np.zeros(48000 * 20, np.float32))
    monkeypatch.setattr(au, "_sd", lambda: sd)
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    x = capture.Recorder().record(stop_event=ev, max_s=10)
    assert len(x) == 0
    monkeypatch.setenv("YQ_MOCK_MIC", "1")
    x = capture.Recorder().record(max_s=5)
    assert len(x) == 3 * 16000


def test_recorder_mic_error_is_spanish(monkeypatch):
    class Broken(FakeSD):
        def InputStream(self, **kw):
            raise OSError("device busy")
    monkeypatch.setattr(au, "_sd", lambda: Broken(np.zeros(10)))
    with pytest.raises(capture.MicError, match="micrófono"):
        capture.Recorder().record(max_s=1)


def test_wav_roundtrip_and_resample():
    x = synthetic_speech(1.0)
    y, r = au.read_wav(au.to_wav_bytes(x))
    assert r == 16000 and np.abs(x - y).max() < 1e-3
    z = au.resample(au.resample(x, 16000, 48000), 48000, 16000)
    assert abs(len(z) - len(x)) <= 1
