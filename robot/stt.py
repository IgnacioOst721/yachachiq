"""Speech to text with faster-whisper on the Pi (CPU, int8). Mock when missing."""
import logging
import time

import numpy as np

import config

log = logging.getLogger("stt")

MOCK_STORY = ("Había una vez una niña que vivía en las montañas con su llama. Un día un cóndor "
              "bajó del cielo hasta el río y le contó una historia sobre el sol y la luna.")


class STT:
    def __init__(self):
        self.model = None
        self.mode = "mock"
        if not config.MOCK:
            try:
                from faster_whisper import WhisperModel   # noqa: F401
                self.mode = "whisper"
            except Exception as e:
                log.warning("faster-whisper not available (%s) -> mock", e)

    def load(self):
        if self.mode != "whisper" or self.model is not None:
            return
        from faster_whisper import WhisperModel
        t0 = time.time()
        kw = dict(device="cpu", compute_type=config.WHISPER_COMPUTE,
                  cpu_threads=int(config.WHISPER_CPU_THREADS))
        try:
            # Load straight from the local cache. Without this the loader asks huggingface.co for
            # the model revision on every start, which at a competition with no internet means a
            # DNS wait before it falls back on its own.
            self.model = WhisperModel(config.WHISPER_MODEL, local_files_only=True, **kw)
        except Exception:
            log.info("whisper '%s' is not in the cache yet: downloading (needs internet)", config.WHISPER_MODEL)
            self.model = WhisperModel(config.WHISPER_MODEL, **kw)
        log.info("whisper '%s' loaded in %.1fs", config.WHISPER_MODEL, time.time() - t0)

    def transcribe(self, audio):
        """audio: float32 mono 16 kHz numpy array (or a wav path). Returns text."""
        if self.mode != "whisper":
            time.sleep(1.0)
            return MOCK_STORY
        self.load()
        kw = {"beam_size": 3, "vad_filter": True}
        if config.WHISPER_LANGUAGE:
            kw["language"] = config.WHISPER_LANGUAGE
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
        segments, info = self.model.transcribe(audio, **kw)
        text = " ".join(s.text.strip() for s in segments).strip()
        log.info("transcribed %d chars (lang %s)", len(text), getattr(info, "language", "?"))
        return text
