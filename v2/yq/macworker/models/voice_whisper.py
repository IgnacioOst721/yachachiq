"""Whisper on the Mac with MLX (mlx-whisper, MIT; weights MIT).

Models (Hugging Face, converted by mlx-community):
  whisper-large-v3        mlx-community/whisper-large-v3-mlx     fp16, 3.1 GB
  whisper-large-v3-turbo  mlx-community/whisper-large-v3-turbo   fp16, 1.6 GB
Heavy imports (mlx, mlx_whisper) happen inside functions only.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np

log = logging.getLogger("yq.voice.whisper")

REPOS = {
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
SIZE_GB = {"whisper-large-v3": 4.7, "whisper-large-v3-turbo": 2.1}   # measured MLX peak on FLEURS (tools/voice_eval.py)


def local_path(repo: str, allow_download: bool = True) -> str:
    """Snapshot folder of `repo`, from the local cache first (works offline)."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        if not allow_download:
            raise
        log.info("downloading %s (needs internet)", repo)
        return snapshot_download(repo)


class WhisperMLX:
    def __init__(self, name: str):
        import mlx.core as mx
        from mlx_whisper.load_models import load_model
        self.name = name
        self.path = local_path(REPOS[name])
        t0 = time.time()
        self.model = load_model(self.path, dtype=mx.float16)
        mx.eval(self.model.parameters())
        log.info("%s loaded in %.1f s", name, time.time() - t0)

    # -- language identification ------------------------------------------------------------
    def detect_language(self, audio: np.ndarray) -> list:
        """[[whisper_code, prob], ...] sorted, from the first 30 s."""
        import mlx.core as mx
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import detect_language
        x = pad_or_trim(np.asarray(audio, dtype=np.float32), N_SAMPLES)
        mel = log_mel_spectrogram(x, n_mels=self.model.dims.n_mels)
        mel = mel[None].astype(mx.float16)
        _, probs = detect_language(self.model, mel)
        p = probs[0] if isinstance(probs, list) else probs
        return sorted(([k, float(v)] for k, v in p.items()), key=lambda kv: -kv[1])

    # -- transcription ---------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, language: Optional[str] = None, prompt: str = "") -> dict:
        """{"text","language","segments":[{start,end,text,confidence}], "confidence"}."""
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
        import mlx.core as mx
        # share our already-loaded weights with mlx_whisper's single-model cache
        ModelHolder.model, ModelHolder.model_path = self.model, self.path
        r = mlx_whisper.transcribe(
            np.asarray(audio, dtype=np.float32), path_or_hf_repo=self.path, language=language,
            initial_prompt=prompt or None, condition_on_previous_text=False, fp16=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8), compression_ratio_threshold=2.4,
            logprob_threshold=-1.0, no_speech_threshold=0.6, hallucination_silence_threshold=None,
            verbose=None)
        mx.clear_cache() if hasattr(mx, "clear_cache") else None
        segs = []
        for s in r.get("segments", []):
            conf = math.exp(min(0.0, float(s.get("avg_logprob", -1.0)))) * (1.0 - float(s.get("no_speech_prob", 0.0)))
            segs.append({"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
                         "text": s["text"].strip(), "confidence": round(conf, 3),
                         "no_speech_prob": float(s.get("no_speech_prob", 0.0))})
        return {"text": r.get("text", "").strip(), "language": r.get("language"), "segments": segs,
                "confidence": weighted_confidence(segs)}


def weighted_confidence(segs: list) -> float:
    tot = sum(max(0.05, s["end"] - s["start"]) for s in segs)
    if not tot:
        return 0.0
    return round(sum(max(0.05, s["end"] - s["start"]) * s["confidence"] for s in segs) / tot, 3)


def load(name: str) -> WhisperMLX:
    return WhisperMLX(name)


def unload(obj: WhisperMLX) -> None:
    try:
        from mlx_whisper.transcribe import ModelHolder
        if ModelHolder.model is getattr(obj, "model", None):
            ModelHolder.model, ModelHolder.model_path = None, None
    except Exception:
        pass
    obj.model = None
