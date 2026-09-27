"""Voice activity detection: Silero VAD (ONNX, MIT) + a small endpointer.

- `SileroVAD`: runs silero_vad.onnx with onnxruntime and numpy only (no torch),
  exactly like the official OnnxWrapper: 512 new samples at 16 kHz plus the
  previous 64 samples of context per call, recurrent state (2, 1, 128).
- `EnergyVAD`: fallback when the model file is missing (logged; much worse).
- `Endpointer`: turns per-frame speech probabilities into "the visitor started
  talking" / "the visitor finished" with pre-roll, minimum speech length and
  trailing-silence stop. Pure Python, so it is tested on synthetic audio.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from . import settings

log = logging.getLogger("yq.voice.vad")

FRAME = 512              # samples per VAD frame at 16 kHz (32 ms)
CONTEXT = 64             # samples of the previous frame the model sees again


class SileroVAD:
    """Streaming Silero VAD. `prob(frame)` -> speech probability 0..1 for 512 samples."""

    name = "silero"

    def __init__(self, path: Optional[Path] = None):
        import onnxruntime as ort
        path = Path(path or settings.vad_model_path())
        if not path.exists():
            raise FileNotFoundError("Silero VAD model not found at %s (run tools/voice_download.py)" % path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != FRAME:
            raise ValueError("Silero VAD needs %d samples at 16 kHz, got %d" % (FRAME, x.shape[1]))
        inp = np.concatenate([self._context, x], axis=1)
        out, state = self.session.run(None, {"input": inp, "state": self._state,
                                             "sr": np.array(settings.SAMPLE_RATE, dtype=np.int64)})
        self._state = state
        self._context = inp[:, -CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVAD:
    """Fallback: speech = RMS clearly above the room's noise floor."""

    name = "energy"

    def __init__(self, factor: float = None, min_rms: float = None):
        self.factor = settings.ENERGY_FACTOR if factor is None else factor
        self.min_rms = settings.ENERGY_MIN_RMS if min_rms is None else min_rms
        self.reset()

    def reset(self) -> None:
        self._noise = collections.deque(maxlen=60)

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))) + 1e-9)
        floor = float(np.percentile(self._noise, 20)) if len(self._noise) >= 8 else self.min_rms / 2
        thr = max(self.min_rms, self.factor * floor)
        if rms < thr:
            self._noise.append(rms)
        return float(np.clip((rms / thr - 0.5), 0.0, 1.0))


def make_vad(path: Optional[Path] = None):
    try:
        return SileroVAD(path)
    except Exception as e:
        log.warning("Silero VAD unavailable (%s): using the energy fallback, endpointing will be worse", e)
        return EnergyVAD()


class Endpointer:
    """Frame-by-frame speech segmentation for one recording.

    feed(frame, prob) returns "start" once when speech begins, "end" once when
    the visitor stopped talking for `silence_s`, "timeout" when nobody spoke for
    `no_speech_s`, else None. `audio()` returns pre-roll + speech + post-roll.
    """

    def __init__(self, threshold: float = None, neg_threshold: float = None, min_speech_s: float = None,
                 silence_s: float = None, preroll_s: float = None, postroll_s: float = None,
                 no_speech_s: float = None, rate: int = settings.SAMPLE_RATE, frame: int = FRAME):
        s = settings
        self.thr = s.VAD_THRESHOLD if threshold is None else threshold
        self.neg = s.VAD_NEG_THRESHOLD if neg_threshold is None else neg_threshold
        fps = rate / float(frame)
        self.min_speech = max(1, int(round((s.VAD_MIN_SPEECH_S if min_speech_s is None else min_speech_s) * fps)))
        self.silence = max(1, int(round((s.VAD_SILENCE_S if silence_s is None else silence_s) * fps)))
        self.preroll = int(round((s.VAD_PREROLL_S if preroll_s is None else preroll_s) * fps))
        self.postroll = int(round((s.VAD_POSTROLL_S if postroll_s is None else postroll_s) * fps))
        ns = s.NO_SPEECH_TIMEOUT_S if no_speech_s is None else no_speech_s
        self.no_speech = int(round(ns * fps)) if ns and ns > 0 else 0
        self.max_gap = max(1, int(round(0.25 * fps)))   # syllable gaps allowed while deciding "speech started"
        self.frame = frame
        self.rate = rate
        self.reset()

    def reset(self) -> None:
        self.state = "waiting"          # waiting -> speaking -> done
        self.n = 0                      # frames seen
        self._pre = collections.deque(maxlen=self.preroll + self.min_speech + self.max_gap * 4 + 8)
        self._cand = 0                  # speech frames in the current candidate onset
        self._cand_len = 0              # frames since the candidate onset began
        self._gap = 0                   # consecutive non-speech frames inside the candidate
        self._frames: list = []
        self._silence_run = 0
        self.speech_frames = 0
        self.start_frame: Optional[int] = None
        self.end_frame: Optional[int] = None
        self.reason = ""

    @property
    def triggered(self) -> bool:
        return self.state in ("speaking", "done") and self.start_frame is not None

    def feed(self, frame: np.ndarray, prob: float) -> Optional[str]:
        if self.state == "done":
            return None
        self.n += 1
        if self.state == "waiting":
            self._pre.append(frame)
            if self._cand == 0:
                if prob >= self.thr:                 # onset
                    self._cand, self._cand_len, self._gap = 1, 1, 0
            else:
                self._cand_len += 1
                if prob >= self.neg:
                    self._cand += 1
                    self._gap = 0
                else:
                    self._gap += 1
                    if self._gap > self.max_gap:     # it was a click / a short noise
                        self._cand, self._cand_len, self._gap = 0, 0, 0
            if self._cand >= self.min_speech:
                keep = min(len(self._pre), self.preroll + self._cand_len)
                self._frames = list(self._pre)[-keep:]
                self.start_frame = self.n - self._cand_len
                self.state = "speaking"
                self.speech_frames = self._cand
                self._silence_run = self._gap
                return "start"
            if self.no_speech and self.n >= self.no_speech:
                self.state, self.reason = "done", "no_speech"
                return "timeout"
            return None
        # speaking
        self._frames.append(frame)
        if prob >= self.neg:
            self._silence_run = 0
            self.speech_frames += 1
        else:
            self._silence_run += 1
            if self._silence_run >= self.silence:
                cut = self._silence_run - self.postroll
                if cut > 0:
                    del self._frames[-cut:]
                self.end_frame = self.n - self._silence_run
                self.state, self.reason = "done", "silence"
                return "end"
        return None

    def finish(self, reason: str = "stopped") -> None:
        """Stop from outside (max length, stop button)."""
        if self.state != "done":
            if self.state == "speaking":
                self.end_frame = self.n
            self.state, self.reason = "done", reason

    def audio(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames).astype(np.float32, copy=False)


def segment(audio: np.ndarray, vad=None, **kw) -> tuple:
    """Run the endpointer over a whole array (tests, files). Returns (speech_audio, endpointer)."""
    vad = vad or make_vad()
    vad.reset()
    ep = Endpointer(**kw)
    x = np.asarray(audio, dtype=np.float32)
    for i in range(0, len(x) - FRAME + 1, FRAME):
        fr = x[i:i + FRAME]
        ev = ep.feed(fr, vad.prob(fr))
        if ev in ("end", "timeout"):
            break
    ep.finish("eof")
    return ep.audio(), ep
