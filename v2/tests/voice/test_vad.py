"""VAD endpointing on synthetic audio: tone, noise, speech-like bursts, silence."""
import numpy as np
import pytest

from yq.voice import vad
from yq.voice.capture import synthetic_speech

SR = 16000


def sil(s, level=0.0005, seed=1):
    return (level * np.random.default_rng(seed).standard_normal(int(s * SR))).astype(np.float32)


def tone(s, f=1000.0, amp=0.3):
    t = np.arange(int(s * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.fixture(scope="module")
def silero():
    pytest.importorskip("onnxruntime")        # voice venv / Jetson; not in requirements-common
    return vad.SileroVAD()


def test_silero_speech_detected_with_preroll_and_trailing_stop(silero):
    speech = synthetic_speech(3.0)
    x = np.concatenate([sil(1.5), speech, sil(4.0)])
    out, ep = vad.segment(x, silero, silence_s=1.0, preroll_s=0.5, postroll_s=0.3, no_speech_s=10)
    assert ep.reason == "silence"
    assert ep.start_frame is not None
    start_s = ep.start_frame * vad.FRAME / SR
    assert 1.3 < start_s < 2.0                         # speech starts at 1.5 s
    dur = len(out) / SR
    assert 3.2 < dur < 4.5, dur                        # 0.5 pre + ~3 speech + 0.3 post
    assert np.abs(out[: int(0.3 * SR)]).max() < 0.05  # pre-roll is the quiet part before speech


def test_silero_ignores_tone_and_white_noise(silero):
    for x in (tone(3.0), (0.2 * np.random.default_rng(3).standard_normal(3 * SR)).astype(np.float32)):
        out, ep = vad.segment(np.concatenate([x, sil(1)]), silero, no_speech_s=3.5)
        assert ep.reason in ("no_speech", "eof") and not ep.triggered


def test_silero_no_speech_timeout(silero):
    out, ep = vad.segment(sil(6.0), silero, no_speech_s=3.0)
    assert ep.reason == "no_speech" and len(out) == 0


def test_short_click_does_not_start_a_story(silero):
    click = np.zeros(int(0.05 * SR), np.float32)
    click[::40] = 0.9
    out, ep = vad.segment(np.concatenate([sil(1), click, sil(3)]), silero, min_speech_s=0.25, no_speech_s=3.5)
    assert not ep.triggered


def test_two_bursts_with_short_pause_stay_one_recording(silero):
    x = np.concatenate([sil(1), synthetic_speech(1.5, seed=1), sil(0.6), synthetic_speech(1.5, seed=2), sil(3)])
    out, ep = vad.segment(x, silero, silence_s=1.2, no_speech_s=5)
    assert ep.reason == "silence" and len(out) / SR > 3.3


def test_endpointer_logic_with_fake_probabilities():
    ep = vad.Endpointer(threshold=0.5, neg_threshold=0.35, min_speech_s=0.1, silence_s=0.32, preroll_s=0.064,
                        postroll_s=0.0, no_speech_s=0)
    fr = np.ones(vad.FRAME, np.float32)
    events = [ep.feed(fr * i, p) for i, p in enumerate([0.1, 0.1, 0.9, 0.9, 0.9, 0.4, 0.9] + [0.0] * 12)]
    assert events.count("start") == 1 and events.count("end") == 1
    assert ep.state == "done" and ep.reason == "silence"
    kept = ep.audio().reshape(-1, vad.FRAME)[:, 0]
    assert list(kept) == [0, 1, 2, 3, 4, 5, 6]         # 2 frames of pre-roll + speech, trailing silence cut


def test_energy_fallback_detects_loud_speech():
    x = np.concatenate([sil(1.5, level=0.002), synthetic_speech(2.0), sil(3, level=0.002)])
    out, ep = vad.segment(x, vad.EnergyVAD(), silence_s=1.0, no_speech_s=5)
    assert ep.triggered and ep.reason == "silence"


def test_make_vad_falls_back_when_model_missing(tmp_path):
    v = vad.make_vad(tmp_path / "missing.onnx")
    assert isinstance(v, vad.EnergyVAD)
