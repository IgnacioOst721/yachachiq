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



def _input_rate(dev):
    """Sample rate to open the microphone at: 16 kHz when the device supports it (what Whisper
    wants), otherwise the device's native rate (cheap USB mics only do 44.1/48 kHz)."""
    try:
        sd.check_input_settings(device=dev, samplerate=config.SAMPLE_RATE, channels=1, dtype="float32")
        return int(config.SAMPLE_RATE)
    except Exception:
        try:
            return int(sd.query_devices(dev, "input")["default_samplerate"])
        except Exception:
            return int(config.SAMPLE_RATE)


def _to_16k(audio, rate):
    """Resample mono float32 audio to config.SAMPLE_RATE (linear; plenty for speech)."""
    if rate == config.SAMPLE_RATE or len(audio) == 0:
        return audio
    n = int(len(audio) * config.SAMPLE_RATE / rate)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


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
        self._fallback = False
        self.no_speech = False

    @staticmethod
    def available():
        """True when an input device exists right now (or we are in mock mode)."""
        if config.MOCK or sd is None:
            return True
        try:
            return any(d["max_input_channels"] > 0 for d in sd.query_devices())
        except Exception:
            return False

    def start(self, on_level=None, on_auto_stop=None):
        self._frames, self._spoke, self._t_last_speech = [], False, None
        self.no_speech = False
        self._fallback = False
        self._auto_stop, self.recording, self._t_start = on_auto_stop, True, time.time()
        self._gen += 1
        gen = self._gen
        if self.mode == "mock" or self._fallback:
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
            silent_start = (not self._spoke) and elapsed > config.NO_SPEECH_SECONDS
            if self.recording and (quiet or silent_start or elapsed > config.MAX_RECORD_SECONDS) and self._auto_stop:
                self.no_speech = silent_start
                self.recording = False
                threading.Thread(target=self._auto_stop, daemon=True).start()

        dev = config.AUDIO_DEVICE or None
        if dev is not None and str(dev).isdigit():
            dev = int(dev)
        try:
            self._rate = _input_rate(dev)
            self._stream = sd.InputStream(samplerate=self._rate, channels=1, dtype="float32",
                                          device=dev, blocksize=1024, callback=cb)
            self._stream.start()
            if self._rate != config.SAMPLE_RATE:
                log.info("microphone at %d Hz, resampling to %d for Whisper", self._rate, config.SAMPLE_RATE)
        except Exception as e:
            # No microphone plugged in (or busy): keep working instead of crashing.
            # The UI chip already says "micrófono: mock"; typed text and sign
            # language still drive the whole pipeline.
            log.warning("no microphone (%s) -> mock recorder for this session", e)
            self._stream = None
            self._fallback = True
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
        return _to_16k(audio, getattr(self, "_rate", config.SAMPLE_RATE))

    def seconds(self):
        return time.time() - self._t_start if self.recording else 0.0


class VoiceTrigger:
    """Keeps the microphone open while the robot is idle and calls on_speech() as soon
    as somebody actually starts talking, so no button, touch screen or keyboard is
    needed. Stops itself while a story is being told (one stream at a time)."""

    def __init__(self, on_speech, on_level=None):
        self.on_speech = on_speech
        self.on_level = on_level
        self.mode = "mock" if (config.MOCK or sd is None) else "mic"
        self._stream = None
        self._loud_since = None
        self.running = False

    def start(self):
        if self.running or self.mode == "mock" or not config.AUTO_LISTEN:
            return False

        def cb(indata, frames, t, status):
            rms = float(np.sqrt(np.mean(indata[:, 0].astype(np.float32) ** 2)) + 1e-9)
            if self.on_level:
                self.on_level(min(1.0, rms * 8))
            now = time.time()
            if rms > config.AUTO_LISTEN_RMS:
                if self._loud_since is None:
                    self._loud_since = now
                elif now - self._loud_since >= config.AUTO_LISTEN_HOLD and self.running:
                    self.running = False          # fire once; the pipeline takes the mic
                    threading.Thread(target=self.on_speech, daemon=True).start()
            else:
                self._loud_since = None

        dev = config.AUDIO_DEVICE or None
        if dev is not None and str(dev).isdigit():
            dev = int(dev)
        try:
            self._stream = sd.InputStream(samplerate=_input_rate(dev), channels=1, dtype="float32",
                                          device=dev, blocksize=2048, callback=cb)
            self._stream.start()
            self.running = True
            log.info("auto-listen armed (speak to start)")
            return True
        except Exception as e:
            log.warning("auto-listen unavailable (%s) - will retry when a microphone appears", e)
            self._stream = None
            return False

    def stop(self):
        self.running = False
        self._loud_since = None
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            finally:
                self._stream = None


def save_wav(audio, path):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(config.SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return str(path)
