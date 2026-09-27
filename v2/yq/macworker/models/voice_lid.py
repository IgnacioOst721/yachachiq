"""Spoken language identification for languages Whisper does not know.

MMS-LID (facebook/mms-lid-4017, CC-BY-NC-4.0 weights, 1B params) through
transformers' Wav2Vec2ForSequenceClassification: 4017 ISO 639-3 labels,
including Southern Quechua varieties (quy, quz, qxp...), Aymara (ayr) and the
Amazonian languages. We restrict its answer to the languages some ASR engine
of ours can transcribe and map labels to canonical codes.
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger("yq.voice.lid")

REPO = "facebook/mms-lid-4017"
SIZE_GB = 2.3          # fp16 weights (1.9 GB) + activations


class MMSLID:
    def __init__(self, repo: str = REPO):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
        try:
            path = snapshot_download(repo, local_files_only=True, allow_patterns=["*.json", "*.safetensors"])
        except Exception:
            path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"])
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        t0 = time.time()
        self.fe = AutoFeatureExtractor.from_pretrained(path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(path, torch_dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        log.info("MMS-LID loaded on %s in %.1f s", self.device, time.time() - t0)

    def predict(self, audio: np.ndarray, allowed_iso3: set = None, top: int = 5, max_s: float = 30.0) -> list:
        """[[iso639_3, prob], ...] best first (renormalised over `allowed_iso3` when given)."""
        import torch
        x = np.asarray(audio, dtype=np.float32)[: int(max_s * 16000)]
        inputs = self.fe(x, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["input_values"] = inputs["input_values"].to(self.dtype)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].float().cpu()
        if allowed_iso3:
            mask = torch.tensor([lab in allowed_iso3 for lab in self.labels])
            logits = logits.masked_fill(~mask, float("-inf"))
        p = torch.softmax(logits, dim=-1)
        vals, idx = p.topk(top)
        if self.device == "mps":
            torch.mps.empty_cache()
        return [[self.labels[int(i)], round(float(v), 4)] for v, i in zip(vals, idx)]


def load() -> MMSLID:
    return MMSLID()
