"""Small audio helpers shared by capture, tts and asr (no heavy imports).

- WAV <-> numpy (16-bit PCM), resampling (windowed-sinc, good enough for speech)
- sounddevice device lookup by name (KAYSUDA speakerphone) with fallback
- the "robot is speaking" flag the recorder uses to mute the microphone
"""
from __future__ import annotations

import io
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Optional, Union

import numpy as np

from . import settings

log = logging.getLogger("yq.voice.audio")

# Set while the robot speaks (TTS playback) and for a short guard time after,
# so the microphone does not record the robot's own voice.
_speaking = threading.Event()
_speaking_until = [0.0]
SPEAK_GUARD_S = 0.35


def set_speaking(on: bool) -> None:
    if on:
        _speaking.set()
    else:
        _speaking.clear()
        _speaking_until[0] = time.monotonic() + SPEAK_GUARD_S


def is_speaking() -> bool:
    return _speaking.is_set() or time.monotonic() < _speaking_until[0]


# --- conversions -------------------------------------------------------------------
def to_wav_bytes(audio: np.ndarray, rate: int = settings.SAMPLE_RATE) -> bytes:
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _read_float_wav(data: bytes) -> tuple:
    """Minimal RIFF reader for IEEE-float WAV (format 3 / extensible), which `wave` rejects."""
    import struct
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if not fmt or raw is None:
        raise ValueError("WAV without fmt/data chunk")
    tag, ch, rate, _, _, bits = fmt
    if bits not in (32, 64):
        raise ValueError("unsupported float WAV (%d bits)" % bits)
    x = np.frombuffer(raw[: len(raw) - len(raw) % (bits // 8)], dtype="<f4" if bits == 32 else "<f8")
    x = x.astype(np.float32)
    if ch > 1:
        x = x[: len(x) - len(x) % ch].reshape(-1, ch).mean(axis=1)
    return x, int(rate)


def read_wav(src: Union[str, Path, bytes]) -> tuple:
    """(float32 mono array, rate) from a PCM (8/16/24/32-bit) or float WAV path or bytes."""
    data = bytes(src) if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    try:
        w = wave.open(io.BytesIO(data), "rb")
    except wave.Error:
        return _read_float_wav(data)
    with w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError("unsupported WAV sample width %d" % width)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), int(rate)


def load_audio(src, rate: int = settings.SAMPLE_RATE) -> np.ndarray:
    """numpy array (assumed `rate`), WAV path or WAV bytes -> float32 mono at `rate`."""
    if isinstance(src, np.ndarray):
        x = src.astype(np.float32, copy=False)
        return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x
    x, r = read_wav(src)
    return resample(x, r, rate)


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited resampling (windowed-sinc low-pass + interpolation)."""
    x = np.asarray(x, dtype=np.float32)
    if src_rate == dst_rate or len(x) == 0:
        return x
    if dst_rate < src_rate:                       # anti-alias before decimating
        cutoff = 0.5 * dst_rate / src_rate * 0.95
        taps = 63
        n = np.arange(taps) - (taps - 1) / 2.0
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
        h /= h.sum()
        x = np.convolve(x, h.astype(np.float32), mode="same")
        ratio = src_rate / dst_rate
        if abs(ratio - round(ratio)) < 1e-9:
            return x[:: int(round(ratio))].astype(np.float32)
    n_out = int(round(len(x) * dst_rate / float(src_rate)))
    t_out = np.arange(n_out) * (src_rate / float(dst_rate))
    return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))) + 1e-12) if len(x) else 0.0


def level(x: np.ndarray) -> float:
    """0..1 meter value: -60 dBFS -> 0, 0 dBFS -> 1."""
    db = 20.0 * np.log10(max(rms(x), 1e-6))
    return float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))


# --- devices -----------------------------------------------------------------------
def _sd():
    import sounddevice as sd
    return sd


def list_devices() -> list:
    try:
        sd = _sd()
        return [dict(index=i, name=d["name"], inputs=d["max_input_channels"], outputs=d["max_output_channels"],
                     rate=d["default_samplerate"]) for i, d in enumerate(sd.query_devices())]
    except Exception as e:
        log.warning("cannot list audio devices: %s", e)
        return []


def find_device(kind: str = "input", explicit="", names: Optional[list] = None) -> Optional[int]:
    """Index of the first device whose name contains one of `names` (case-insensitive).

    `explicit` (index or name) wins. None = let PortAudio use the system default.
    """
    key = "inputs" if kind == "input" else "outputs"
    devs = [d for d in list_devices() if d[key] > 0]
    if explicit not in ("", None):
        if str(explicit).isdigit():
            return int(explicit)
        for d in devs:
            if str(explicit).lower() in d["name"].lower():
                return d["index"]
        log.warning("%s device %r not found; using the default", kind, explicit)
        return None
    for want in names or []:
        for d in devs:
            if want.lower() in d["name"].lower():
                return d["index"]
    return None


def input_rate(device: Optional[int]) -> int:
    """16 kHz when the device accepts it, else its native rate (we resample)."""
    sd = _sd()
    try:
        sd.check_input_settings(device=device, samplerate=settings.SAMPLE_RATE, channels=1, dtype="float32")
        return settings.SAMPLE_RATE
    except Exception:
        return int(sd.query_devices(device, "input")["default_samplerate"])


def save_wav(audio: np.ndarray, path, rate: int = settings.SAMPLE_RATE) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(to_wav_bytes(audio, rate))
    return str(path)
