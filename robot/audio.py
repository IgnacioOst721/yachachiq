"""Microphone recording with a live level meter and silence auto-stop.

Recorder().start(on_level, on_auto_stop) ... stop() -> float32 mono 16 kHz array
Mock mode (no sounddevice / config.MOCK): records nothing and "auto-stops" after
~6 s, so the UI flow can be tried on any computer.
"""
import logging
import threading
import time
import wave

import numpy as np

import config

log = logging.getLogger("audio")

try:
    import sounddevice as sd
except Exception:               # pragma: no cover
    sd = None


class Recorder:
    def __init__(self):
        self.mode = "mock" if (config.MOCK or sd is None) else "mic"
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()
        self._t_start = 0.0
        self._t_last_speech = None
        self._spoke = False
        self._timer = None
        self._auto_stop = None
        self.recording = False
        self._gen = 0                      # session counter: an old mock session must not stop a new one

    def start(self, on_level=None, on_auto_stop=None):
        self._frames, self._spoke, self._t_last_speech = [], False, None
        self._auto_stop, self.recording, self._t_start = on_auto_stop, True, time.time()
        self._gen += 1
        gen = self._gen
        if self.mode == "mock":
            def fake():
                t0 = time.time()
                while self.recording and self._gen == gen and time.time() - t0 < 6.0:
                    if on_level:
                        on_level(0.05 + 0.04 * abs(np.sin(time.time() * 6)))
                    time.sleep(0.1)
                if self.recording and self._gen == gen and self._auto_stop:
                    self._auto_stop()
            self._timer = threading.Thread(target=fake, daemon=True)
            self._timer.start()
            return True

        def cb(indata, frames, t, status):
            chunk = indata[:, 0].astype(np.float32).copy()
            rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-9)
            with self._lock:
                self._frames.append(chunk)
            if on_level:
                on_level(min(1.0, rms * 8))
            now = time.time()
            if rms > config.SILENCE_RMS:
                self._spoke, self._t_last_speech = True, now
            elapsed = now - self._t_start
            quiet = self._spoke and self._t_last_speech and (now - self._t_last_speech) > config.SILENCE_SECONDS
            if self.recording and (quiet or elapsed > config.MAX_RECORD_SECONDS) and self._auto_stop:
                self.recording = False
                threading.Thread(target=self._auto_stop, daemon=True).start()

        dev = config.AUDIO_DEVICE or None
        if dev is not None and str(dev).isdigit():
            dev = int(dev)
        try:
            self._stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                                          device=dev, blocksize=1024, callback=cb)
            self._stream.start()
        except Exception as e:
            # No microphone plugged in (or busy): keep working instead of crashing.
            # The UI chip already says "micrófono: mock"; typed text and sign
            # language still drive the whole pipeline.
            log.warning("no microphone (%s) -> mock recorder", e)
            self.mode = "mock"
            self._stream = None
            return self.start(on_level=on_level, on_auto_stop=on_auto_stop)
        return True

    def stop(self):
        self.recording = False
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            finally:
                self._stream = None
        with self._lock:
            audio = np.concatenate(self._frames) if self._frames else np.zeros(config.SAMPLE_RATE, np.float32)
        return audio

    def seconds(self):
        return time.time() - self._t_start if self.recording else 0.0


def save_wav(audio, path):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(config.SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return str(path)
