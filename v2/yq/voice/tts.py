"""Text to speech on the Jetson (CONTRACTS.md §6).

    from yq.voice import tts
    tts.say("Hola, cuéntame tu historia", "spa_Latn")          # returns at once, speaks in background
    tts.say("Allinllachu", "quy_Latn", wait=True)             # blocks until done
    tts.stop()

Order: a local Piper voice for the language (best quality first, files in
MODELS_DIR/piper, downloaded beforehand by tools/voice_download.py) -> the Mac's
POST /tts (MMS-TTS, ~1100 languages incl. Quechua and Aymara) -> False (silent).
While speaking, the microphone is muted (yq.voice.audio.is_speaking()).
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from yq.common import config, languages

from . import audio as au
from . import settings

log = logging.getLogger("yq.voice.tts")

_voices: dict = {}
_lock = threading.Lock()
_play_lock = threading.Lock()
_stop = threading.Event()
_thread: Optional[threading.Thread] = None
_proc: Optional[subprocess.Popen] = None
last_engine = ""


# --- Piper voices -----------------------------------------------------------------------
def voice_files(key: str) -> tuple:
    d = settings.piper_dir()
    return d / (key + ".onnx"), d / (key + ".onnx.json")


def candidate_voices(lang: str) -> list:
    """Piper voice keys for `lang`, best first (settings override, then quality)."""
    L = languages.get(lang)
    if not L:
        return []
    keys = []
    ov = settings.piper_overrides().get(L.code)
    if ov:
        keys.append(ov)
    keys += [v["key"] for v in L.engines.get("piper", []) if v["key"] not in keys]
    return keys


def local_voice(lang: str) -> Optional[str]:
    for key in candidate_voices(lang):
        onnx, js = voice_files(key)
        if onnx.exists() and js.exists():
            return key
    return None


def ensure_voice(lang: str, key: Optional[str] = None) -> Optional[str]:
    """Download the best Piper voice for `lang` (needs internet; used by the download tool)."""
    import requests
    L = languages.get(lang)
    if not L:
        return None
    voices = {v["key"]: v for v in L.engines.get("piper", [])}
    key = key or next((k for k in candidate_voices(lang) if k in voices), None)
    if not key:
        return None
    onnx, js = voice_files(key)
    onnx.parent.mkdir(parents=True, exist_ok=True)
    rel = voices[key]["onnx"]
    for url, dest in ((settings.PIPER_BASE_URL + rel, onnx), (settings.PIPER_BASE_URL + rel + ".json", js)):
        if dest.exists() and dest.stat().st_size > 0:
            continue
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
    return key


def _piper(key: str):
    with _lock:
        if key not in _voices:
            from piper import PiperVoice
            onnx, js = voice_files(key)
            _voices[key] = PiperVoice.load(onnx, config_path=js, use_cuda=bool(settings.PIPER_USE_CUDA))
        return _voices[key]


def synthesize_piper(text: str, key: str) -> tuple:
    """(float32 audio, sample_rate) with the Piper voice `key`."""
    from piper import SynthesisConfig
    voice = _piper(key)
    cfg = SynthesisConfig(length_scale=float(settings.TTS_LENGTH_SCALE))
    chunks = [c.audio_float_array for c in voice.synthesize(text, syn_config=cfg)]
    rate = int(json.loads(voice_files(key)[1].read_text())["audio"]["sample_rate"])
    return (np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)), rate


def synthesize_mac(text: str, lang: str) -> tuple:
    from yq.common.macclient import client
    wav = client()._request("POST", "/tts", json={"text": text, "lang": lang}, timeout=settings.TTS_MAC_TIMEOUT_S)
    if not wav.ok:
        raise RuntimeError("Mac /tts HTTP %d: %s" % (wav.status_code, wav.text[:200]))
    return au.read_wav(wav.content)


def synthesize(text: str, lang: str) -> tuple:
    """(audio, rate, engine) using the best available engine; raises when none."""
    L = languages.get(lang)
    code = L.code if L else lang
    key = local_voice(code)
    errors = []
    if key:
        try:
            x, r = synthesize_piper(text, key)
            return x, r, "piper:" + key
        except Exception as e:
            errors.append("piper %s: %s" % (key, e))
    if not config.mock("mac"):
        try:
            x, r = synthesize_mac(text, code)
            return x, r, "mac"
        except Exception as e:
            errors.append("mac: %s" % e)
    raise RuntimeError("no TTS for %s (%s)" % (code, "; ".join(errors) or "no voice"))


# --- playback -------------------------------------------------------------------------
def _play(x: np.ndarray, rate: int) -> None:
    global _proc
    au.set_speaking(True)
    try:
        if config.mock("speaker"):
            t_end = time.monotonic() + len(x) / float(rate)
            while time.monotonic() < t_end and not _stop.is_set():
                time.sleep(0.05)
            return
        if settings.AUDIO_PLAYER:
            _proc = subprocess.Popen([settings.AUDIO_PLAYER, "-q", "-"], stdin=subprocess.PIPE)
            try:
                _proc.stdin.write(au.to_wav_bytes(x, rate))
                _proc.stdin.close()
            except BrokenPipeError:
                pass
            while _proc.poll() is None:
                if _stop.is_set():
                    _proc.terminate()
                    break
                time.sleep(0.05)
            return
        sd = au._sd()
        dev = au.find_device("output", settings.SPEAKER_DEVICE, settings.SPEAKER_NAMES)
        try:
            sd.check_output_settings(device=dev, samplerate=rate, channels=1, dtype="float32")
        except Exception:        # e.g. a USB speakerphone that only plays 48 kHz
            dev_rate = int(sd.query_devices(dev, "output")["default_samplerate"])
            x, rate = au.resample(x, rate, dev_rate), dev_rate
        sd.play(x, rate, device=dev)
        t_end = time.monotonic() + len(x) / float(rate) + 0.5
        while time.monotonic() < t_end:
            if _stop.is_set():
                sd.stop()
                break
            time.sleep(0.05)
        sd.wait()
    finally:
        au.set_speaking(False)


def say(text: str, lang: str = "spa_Latn", wait: bool = False) -> bool:
    """Speak `text` in `lang`. False (and silence) when no engine can speak it."""
    global _thread, last_engine
    text = (text or "").strip()
    if not text:
        return False
    if config.mock("tts"):
        last_engine = "mock"
        return True
    try:
        x, rate, eng = synthesize(text, lang)
    except Exception as e:
        log.info("tts unavailable: %s", e)
        last_engine = "none"
        return False
    last_engine = eng
    stop()
    _stop.clear()

    def run():
        with _play_lock:
            try:
                _play(x, rate)
            except Exception as e:
                log.warning("playback failed: %s", e)
    _thread = threading.Thread(target=run, daemon=True, name="tts-play")
    _thread.start()
    if wait:
        _thread.join()
    return True


def stop() -> None:
    _stop.set()
    th = _thread
    if th is not None and th.is_alive() and th is not threading.current_thread():
        th.join(timeout=2.0)


def speaking() -> bool:
    return au.is_speaking()
