"""Speech to text on the Jetson (CONTRACTS.md §6).

    from yq.voice import asr
    t = asr.transcribe(audio, lang="auto")      # -> yq.common.contracts.Transcript

Order, never hanging (every step has a timeout):
 1. Mac worker POST /asr (Whisper large-v3 / Omnilingual ASR, best accuracy).
 2. Jetson GPU: faster-whisper large-v3-turbo (CTranslate2 built with CUDA).
 3. Jetson CPU: faster-whisper "small" int8 (slow but works anywhere).
 4. Mock (only with YQ_MOCK / YQ_MOCK_ASR): a fixed Spanish story.
When every real engine fails the result is an empty Transcript with
engine="none" and confidence 0, so the kiosk asks the visitor to type or retry
instead of inventing a story.

Local Whisper cannot hear Quechua, Aymara or Amazonian languages: for those the
Mac is required; if it is down the local model still tries and the result is
marked with low confidence (the visitor must confirm the text).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional

import numpy as np

from yq.common import config, languages
from yq.common.contracts import Segment, Transcript, from_dict, to_dict

from . import audio as au
from . import settings

log = logging.getLogger("yq.voice.asr")

MOCK_STORY = ("Había una vez una niña que vivía en las montañas con su llama. Un día un cóndor bajó "
              "del cielo hasta el río y le contó una historia sobre el sol y la luna.")

_models: dict = {}
_model_lock = threading.Lock()


def _duration(x: np.ndarray) -> float:
    return round(len(x) / float(settings.SAMPLE_RATE), 2)


# --- 1. Mac ---------------------------------------------------------------------------
def _mac(x: np.ndarray, lang: str, prompt: str) -> Transcript:
    from yq.common.macclient import client
    data = {"lang": lang or "auto"}
    if prompt:
        data["prompt"] = prompt
    d = client().post_files("/asr", {"audio": ("audio.wav", au.to_wav_bytes(x), "audio/wav")}, data=data,
                            timeout=settings.ASR_MAC_TIMEOUT_S)
    return from_dict(Transcript, d)


# --- 2/3. faster-whisper on the Jetson ------------------------------------------------------
def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _fw_model(kind: str):
    """kind: "gpu" (large-v3-turbo on CUDA) or "cpu" (small int8)."""
    with _model_lock:
        if kind in _models:
            return _models[kind]
        from faster_whisper import WhisperModel
        if kind == "gpu":
            name, kw = settings.ASR_JETSON_MODEL, dict(device="cuda", compute_type=settings.ASR_JETSON_COMPUTE)
        else:
            name, kw = settings.ASR_CPU_MODEL, dict(device="cpu", compute_type=settings.ASR_CPU_COMPUTE,
                                                    cpu_threads=int(settings.ASR_CPU_THREADS))
        root = str(settings.whisper_cache())
        t0 = time.time()
        try:   # offline first: never wait for huggingface.co at the competition
            m = WhisperModel(name, download_root=root, local_files_only=True, **kw)
        except Exception:
            log.info("faster-whisper %s not cached in %s: downloading (needs internet)", name, root)
            m = WhisperModel(name, download_root=root, **kw)
        log.info("faster-whisper %s (%s) loaded in %.1f s", name, kind, time.time() - t0)
        _models[kind] = m
        return m


def segments_confidence(segs: list) -> float:
    """Duration-weighted exp(avg_logprob) x (1 - no_speech_prob), 0..1."""
    tot, acc = 0.0, 0.0
    for s in segs:
        d = max(0.05, float(s["end"]) - float(s["start"]))
        p = math.exp(min(0.0, float(s.get("avg_logprob", -1.0)))) * (1.0 - float(s.get("no_speech_prob", 0.0)))
        tot += d
        acc += d * p
    return round(acc / tot, 3) if tot else 0.0


def _faster_whisper(x: np.ndarray, lang: str, prompt: str, kind: str) -> Transcript:
    m = _fw_model(kind)
    L = languages.get(lang) if lang and lang != "auto" else None
    wcode = L.whisper_code if L else None
    penalty = 1.0
    if L is not None and not wcode:
        penalty = 0.4       # Whisper does not know this language: let it guess, flag low confidence
    segs, info = m.transcribe(x, language=wcode, initial_prompt=prompt or None, beam_size=int(settings.ASR_BEAM),
                              vad_filter=False, condition_on_previous_text=False)
    segs = [dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip(), avg_logprob=s.avg_logprob,
                 no_speech_prob=s.no_speech_prob) for s in segs]
    text = " ".join(s["text"] for s in segs).strip()
    cands = []
    if info.all_language_probs:
        for code, p in info.all_language_probs[:5]:
            c = languages.resolve(code)
            if c:
                cands.append([c, round(float(p), 4)])
    detected = languages.resolve(info.language) or (L.code if L else "")
    conf = segments_confidence(segs) * penalty
    name = settings.ASR_JETSON_MODEL if kind == "gpu" else settings.ASR_CPU_MODEL
    name = name.rsplit("/", 1)[-1].replace("faster-whisper-", "")      # "large-v3-turbo", "small"
    return Transcript(text=text, lang=(L.code if L and wcode else detected), engine="whisper-%s@jetson-%s" % (name, kind),
                      confidence=round(conf, 3), duration_s=_duration(x),
                      segments=[to_dict(Segment(s["start"], s["end"], s["text"],
                                                round(segments_confidence([s]), 3))) for s in segs],
                      lang_candidates=cands)


def _with_timeout(fn, timeout: float, *args):
    """Run fn(*args) in a daemon thread; raise TimeoutError when it takes too long."""
    box: dict = {}

    def run():
        try:
            box["v"] = fn(*args)
        except BaseException as e:   # noqa: BLE001 (re-raised in the caller)
            box["e"] = e
    th = threading.Thread(target=run, daemon=True, name="asr-local")
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError("%s took more than %.0f s" % (getattr(fn, "__name__", "asr"), timeout))
    if "e" in box:
        raise box["e"]
    return box["v"]


def engines_order() -> list:
    order = [] if config.mock("mac") else ["mac"]
    if settings.ASR_LOCAL and not config.mock("asr"):
        if cuda_available():
            order.append("jetson-gpu")
        order.append("jetson-cpu")
    return order


def transcribe(audio, lang: str = "auto", prompt: str = "") -> Transcript:
    """audio: float32 16 kHz numpy array, WAV path or WAV bytes. lang: canonical code or "auto"."""
    x = au.load_audio(audio)
    if lang and lang != "auto":
        lang = languages.resolve(lang) or lang
    if config.mock("asr"):
        time.sleep(0.3)
        return Transcript(text=MOCK_STORY, lang="spa_Latn", engine="mock", confidence=1.0,
                          duration_s=_duration(x))
    if len(x) < int(0.2 * settings.SAMPLE_RATE):
        return Transcript(text="", lang=lang if lang != "auto" else "", engine="none", confidence=0.0,
                          duration_s=_duration(x))
    errors = []
    for eng in engines_order():
        try:
            if eng == "mac":
                t = _mac(x, lang, prompt)
            else:
                kind = "gpu" if eng == "jetson-gpu" else "cpu"
                t = _with_timeout(_faster_whisper, settings.ASR_LOCAL_TIMEOUT_S, x, lang, prompt, kind)
            log.info("asr %s: %d chars, lang %s, conf %.2f", t.engine, len(t.text), t.lang, t.confidence)
            return t
        except Exception as e:
            log.warning("asr engine %s failed: %s", eng, e)
            errors.append("%s: %s" % (eng, e))
    log.error("every ASR engine failed: %s", "; ".join(errors))
    return Transcript(text="", lang=lang if lang != "auto" else "", engine="none", confidence=0.0,
                      duration_s=_duration(x))


def detect_language(audio) -> list:
    """[[code, prob], ...] from the Mac /lid (empty when the Mac is unavailable)."""
    if config.mock("asr") or config.mock("mac"):
        return [["spa_Latn", 1.0]]
    try:
        from yq.common.macclient import client
        x = au.load_audio(audio)
        d = client().post_files("/lid", {"audio": ("audio.wav", au.to_wav_bytes(x), "audio/wav")},
                                timeout=settings.ASR_MAC_TIMEOUT_S)
        return d.get("candidates", [])
    except Exception as e:
        log.warning("lid failed: %s", e)
        return []


def preload(kind: Optional[str] = None) -> bool:
    """Load the local model now (call at boot so the first story is not slow)."""
    try:
        _fw_model(kind or ("gpu" if cuda_available() else "cpu"))
        return True
    except Exception as e:
        log.warning("local ASR preload failed: %s", e)
        return False
