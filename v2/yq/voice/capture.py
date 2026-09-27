"""Record one story from the microphone with automatic start/stop (CONTRACTS.md §6).

    from yq.voice.capture import Recorder
    audio = Recorder().record(on_level=lambda x: ..., max_s=90.0, stop_event=ev)
    # float32 mono 16 kHz numpy array (empty when nobody spoke)

- Device: the KAYSUDA USB speakerphone found by name (settings.MIC_NAMES), else
  the system default input. 16 kHz when the device allows it, else its native
  rate resampled to 16 kHz.
- Endpointing: Silero VAD (ONNX) frames of 32 ms; keeps 0.5 s of pre-roll,
  stops after 1.8 s of silence, gives up after 12 s without speech.
- While the robot speaks (tts), microphone frames are discarded.
- Mock (YQ_MOCK=1 or YQ_MOCK_MIC=1): returns 3 s of synthetic speech-like audio.

Test the microphone:  python -m yq.voice.capture --list
                      python -m yq.voice.capture --test   (speak, it saves test.wav)
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from yq.common import config

from . import audio as au
from . import settings
from .vad import FRAME, Endpointer, make_vad

log = logging.getLogger("yq.voice.capture")


class MicError(RuntimeError):
    """The microphone could not be opened (Spanish message for the kiosk)."""


def synthetic_speech(seconds: float = 3.0, rate: int = settings.SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """Speech-like test signal: voiced bursts (harmonics of ~140 Hz, syllable-rate envelope) + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    voiced = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None) ** 0.6       # ~4 syllables per second
    x = 0.25 * voiced * env + 0.003 * rng.standard_normal(len(t))
    return x.astype(np.float32)


class Recorder:
    def __init__(self, device=None, vad=None):
        self.mock = config.mock("mic")
        self._device_arg = device
        self._vad = vad
        self.last_reason = ""          # "silence" | "no_speech" | "max" | "stopped" | "mock"
        self.last_device = ""
        self.last_vad = ""

    @staticmethod
    def available() -> bool:
        if config.mock("mic"):
            return True
        return any(d["inputs"] > 0 for d in au.list_devices())

    # ------------------------------------------------------------------------------
    def record(self, on_level: Optional[Callable[[float], None]] = None, max_s: float = None,
               stop_event: Optional[threading.Event] = None) -> np.ndarray:
        max_s = float(max_s if max_s is not None else settings.MAX_RECORD_S)
        if self.mock:
            return self._record_mock(on_level, max_s, stop_event)
        sd = au._sd()
        dev = self._device_arg
        if dev is None:
            dev = au.find_device("input", settings.MIC_DEVICE, settings.MIC_NAMES)
        try:
            rate = au.input_rate(dev)
            self.last_device = sd.query_devices(dev, "input")["name"]
        except Exception as e:
            raise MicError("No pude abrir el micrófono (%s). Revisa que esté conectado." % e)
        vad = self._vad or make_vad()
        vad.reset()
        self.last_vad = getattr(vad, "name", "?")
        ep = Endpointer()
        q: "queue.Queue[np.ndarray]" = queue.Queue()
        block = max(256, int(rate * settings.MIC_BLOCK_MS / 1000))

        def callback(indata, frames, t, status):   # PortAudio thread: copy and hand over only
            q.put(indata[:, 0].astype(np.float32, copy=True))

        try:
            stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=dev,
                                    blocksize=block, callback=callback)
            stream.start()
        except Exception as e:
            raise MicError("No pude abrir el micrófono (%s). Revisa que esté conectado, "
                           "o cuenta tu historia en lengua de señas." % e)
        log.info("recording from %r at %d Hz (vad=%s)", self.last_device, rate, self.last_vad)
        pending = np.zeros(0, dtype=np.float32)
        t0 = time.monotonic()
        reason = "stopped"
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    reason = "stopped"
                    break
                if time.monotonic() - t0 > max_s:
                    reason = "max"
                    break
                try:
                    chunk = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if rate != settings.SAMPLE_RATE:
                    chunk = au.resample(chunk, rate, settings.SAMPLE_RATE)
                if au.is_speaking():          # the robot is talking: ignore our own voice
                    pending = np.zeros(0, dtype=np.float32)
                    if on_level:
                        on_level(0.0)
                    continue
                pending = np.concatenate([pending, chunk])
                ev = None
                while len(pending) >= FRAME:
                    fr, pending = pending[:FRAME], pending[FRAME:]
                    ev = ep.feed(fr, vad.prob(fr)) or ev
                    if on_level:
                        on_level(au.level(fr))
                    if ep.state == "done":
                        break
                if ep.state == "done":
                    reason = ep.reason
                    break
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        ep.finish(reason)
        self.last_reason = ep.reason
        out = ep.audio() if ep.triggered else np.zeros(0, dtype=np.float32)
        log.info("recorded %.1f s of speech (%s)", len(out) / settings.SAMPLE_RATE, self.last_reason)
        return out

    def _record_mock(self, on_level, max_s, stop_event) -> np.ndarray:
        x = synthetic_speech(min(3.0, max_s))
        step = FRAME
        for i in range(0, len(x), step):
            if stop_event is not None and stop_event.is_set():
                break
            if on_level:
                on_level(au.level(x[i:i + step]))
            time.sleep(step / settings.SAMPLE_RATE / 4)      # 4x real time
        self.last_reason = "mock"
        return x


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Microphone test for Yachachiq (voice)")
    ap.add_argument("--list", action="store_true", help="list audio devices")
    ap.add_argument("--test", action="store_true", help="record one utterance and save it")
    ap.add_argument("--out", default="test.wav")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.list or not a.test:
        chosen = au.find_device("input", settings.MIC_DEVICE, settings.MIC_NAMES)
        for d in au.list_devices():
            mark = "  <== micrófono elegido" if d["index"] == chosen else ""
            print("%2d  in=%d out=%d  %6.0f Hz  %s%s" % (d["index"], d["inputs"], d["outputs"], d["rate"], d["name"], mark))
        if chosen is None:
            print("(ningún nombre de MIC_NAMES coincide: se usa el micrófono por defecto)")
    if a.test:
        dev = int(a.device) if a.device and a.device.isdigit() else (au.find_device("input", a.device) if a.device else None)
        rec = Recorder(device=dev)
        print("Habla ahora (se detiene solo cuando te callas)...")

        def meter(v):
            print("\r[%-40s]" % ("#" * int(v * 40)), end="", flush=True)
        x = rec.record(on_level=meter, max_s=30)
        print()
        au.save_wav(x, a.out)
        print("%.1f s guardados en %s  (motivo: %s, micrófono: %s, vad: %s)" %
              (len(x) / settings.SAMPLE_RATE, a.out, rec.last_reason, rec.last_device, rec.last_vad))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
