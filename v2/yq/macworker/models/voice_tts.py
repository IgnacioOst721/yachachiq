"""Text to speech on the Mac: Meta MMS-TTS (VITS, facebook/mms-tts-<iso3>, ~1100
languages including Quechua varieties, Aymara, Asháninka, Awajún, Shipibo...).
Weights are CC-BY-NC-4.0 (flagged in docs/licenses_voice.md).

Some MMS voices expect romanized input (`tokenizer.is_uroman`); those need the
`uroman` package (pip install uroman). Latin-script languages do not.
Piper voices (when installed on the Mac) are used for languages MMS lacks.
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger("yq.voice.tts")

SIZE_GB = 0.35   # 145 MB weights, ~36M params fp32 + activations


_UROMAN = None


def _uroman():
    """uroman (MIT-style, acknowledgement requested) for MMS voices trained on romanized text."""
    global _UROMAN
    if _UROMAN is None:
        try:
            import uroman as ur
        except ImportError:
            raise RuntimeError("this MMS voice needs romanized text: pip install uroman")
        _UROMAN = ur.Uroman()
    return _UROMAN


class MMSTTS:
    def __init__(self, iso3: str):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, VitsModel
        repo = "facebook/mms-tts-" + iso3
        try:
            path = snapshot_download(repo, local_files_only=True)
        except Exception:
            path = snapshot_download(repo)
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = VitsModel.from_pretrained(path).eval()        # CPU: small model, MPS gains nothing
        self.rate = int(self.model.config.sampling_rate)
        self.iso3 = iso3
        self.torch = torch
        log.info("MMS-TTS %s loaded in %.1f s", iso3, time.time() - t0)

    def _prepare(self, text: str) -> str:
        if getattr(self.tokenizer, "is_uroman", False):
            text = _uroman().romanize_string(text, lcode=self.iso3)
        return text

    def synthesize(self, text: str, seed: int = 0) -> tuple:
        """(float32 audio, rate). Sentence by sentence so long texts stay stable."""
        import re
        torch = self.torch
        torch.manual_seed(seed)
        pieces = [p for p in re.split(r"(?<=[.!?;:])\s+", text.strip()) if p.strip()] or [text]
        out = []
        pause = np.zeros(int(0.25 * self.rate), dtype=np.float32)
        for p in pieces:
            inputs = self.tokenizer(self._prepare(p), return_tensors="pt")
            if inputs["input_ids"].shape[-1] == 0:
                continue
            with torch.inference_mode():
                wav = self.model(**inputs).waveform[0].float().cpu().numpy()
            out += [wav.astype(np.float32), pause]
        return (np.concatenate(out) if out else np.zeros(0, np.float32)), self.rate


def load(iso3: str) -> MMSTTS:
    return MMSTTS(iso3)


class PiperMac:
    """Piper voice on the Mac (only when piper-tts and the voice file are present)."""

    def __init__(self, key: str):
        from yq.voice import tts as jtts
        self.key = key
        self._jtts = jtts
        if not jtts.voice_files(key)[0].exists():
            raise FileNotFoundError("piper voice %s not downloaded" % key)

    def synthesize(self, text: str) -> tuple:
        return self._jtts.synthesize_piper(text, self.key)
