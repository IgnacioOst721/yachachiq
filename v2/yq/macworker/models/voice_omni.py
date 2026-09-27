"""Meta Omnilingual ASR (Apache-2.0, 1600+ languages) on the Mac, in a child process.

Why a child process: fairseq2's native library (fairseq2n 0.6 macOS wheel) is
linked against Homebrew's libsndfile (/opt/homebrew/...). This Mac has no
Homebrew, so the child is started with DYLD_FALLBACK_LIBRARY_PATH pointing to
the libsndfile that the `soundfile` wheel already ships. As a bonus, unloading
= killing the child, which really returns all the memory (MPS caches included).

Parent side:  OmniASR(card).transcribe(audio16k, lang="quy_Latn") -> {"text","confidence",...}
Child side:   python -m yq.macworker.models.voice_omni serve --card omniASR_CTC_1B_v2
Protocol: one JSON object per line on stdin/stdout ({"wav": path, "lang": code}).
Audio longer than 38 s is split at the quietest point (the models accept < 40 s).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("yq.voice.omni")

CARDS = {   # our name -> (fairseq2 model card, size_gb for the model manager)
    "omniasr-ctc-300m": ("omniASR_CTC_300M_v2", 2.0),       # README: ~2 GiB (not measured here)
    "omniasr-ctc-1b": ("omniASR_CTC_1B_v2", 4.6),           # measured peak RSS + MPS on FLEURS/Quechua sets
    "omniasr-llm-300m": ("omniASR_LLM_300M_v2", 5.0),       # README: ~5 GiB (not measured here)
    "omniasr-llm-1b": ("omniASR_LLM_1B_v2", 5.9),           # measured RSS + MPS (smoke test)
}
MAX_CHUNK_S = 38.0
SR = 16000


def sndfile_dir(python: str = "") -> Path:
    """Folder containing a file named libsndfile.1.dylib (symlink to the `soundfile` wheel's copy)."""
    from yq.common import config
    d = Path(config.DATA_DIR) / "cache" / "voice_libs"
    target = d / "libsndfile.1.dylib"
    if target.exists():
        return d
    libs = []
    if python:      # the venv that will run the child
        venv = Path(python).absolute().parents[1] if Path(python).exists() else None   # venv root, not the symlink target
        if venv:
            libs = sorted(venv.glob("lib/python3*/site-packages/_soundfile_data/libsndfile*.dylib"))
    if not libs:
        try:
            import soundfile
            libs = sorted((Path(soundfile.__file__).parent / "_soundfile_data").glob("libsndfile*.dylib"))
        except ImportError:
            pass
    if not libs:
        raise RuntimeError("libsndfile not found: install `soundfile` in the voice venv")
    d.mkdir(parents=True, exist_ok=True)
    target.symlink_to(libs[0])
    return d


def split_long(x: np.ndarray, max_s: float = MAX_CHUNK_S, sr: int = SR) -> list:
    """Split at the quietest 100 ms window in the last third of each chunk."""
    out, i, n = [], 0, int(max_s * sr)
    while len(x) - i > n:
        lo, hi = i + int(n * 0.66), i + n
        win = int(0.1 * sr)
        seg = x[lo:hi]
        e = np.convolve(seg.astype(np.float64) ** 2, np.ones(win), mode="valid")
        cut = lo + int(np.argmin(e)) + win // 2
        out.append(x[i:cut])
        i = cut
    out.append(x[i:])
    return [c for c in out if len(c) > int(0.1 * sr)]


# --------------------------------------------------------------------------- parent
class OmniASR:
    def __init__(self, name: str = "omniasr-ctc-1b", device: str = "auto", load_timeout: float = 900.0):
        self.name = name
        self.card = CARDS[name][0]
        from yq.voice import settings
        python = settings.voice_python()
        env = dict(os.environ)
        if sys.platform == "darwin":
            prev = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            env["DYLD_FALLBACK_LIBRARY_PATH"] = str(sndfile_dir(python)) + (":" + prev if prev else "")
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        v2 = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = v2 + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.proc = subprocess.Popen([python, "-m", "yq.macworker.models.voice_omni", "serve",
                                      "--card", self.card, "--device", device],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                                     text=True, bufsize=1, env=env, cwd=v2)
        self._lock = threading.Lock()
        ready = self._read(load_timeout)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError("omni child failed to start: %s" % ready.get("error", ready))
        self.info = ready

    def _read(self, timeout: float) -> dict:
        box: dict = {}

        def rd():
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    box["eof"] = True
                    return
                line = line.strip()
                if line.startswith("{"):
                    box["msg"] = json.loads(line)
                    return
        th = threading.Thread(target=rd, daemon=True)
        th.start()
        th.join(timeout)
        if "msg" in box:
            return box["msg"]
        if th.is_alive():
            self.close()
            raise TimeoutError("omni child did not answer in %.0f s" % timeout)
        raise RuntimeError("omni child exited (code %s)" % self.proc.poll())

    def transcribe(self, audio: np.ndarray, lang: Optional[str] = None, timeout: float = 300.0) -> dict:
        x = np.asarray(audio, dtype=np.float32)
        texts, confs, weights, t0, mem = [], [], [], time.time(), 0.0
        with self._lock:
            for chunk in split_long(x):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                    path = fh.name
                try:
                    from yq.voice.audio import save_wav
                    save_wav(chunk, path)
                    self.proc.stdin.write(json.dumps({"wav": path, "lang": lang}) + "\n")
                    self.proc.stdin.flush()
                    r = self._read(timeout)
                finally:
                    Path(path).unlink(missing_ok=True)
                if "error" in r:
                    raise RuntimeError("omni: %s" % r["error"])
                texts.append(r["text"].strip())
                mem = max(mem, float(r.get("mem_gb") or 0.0))
                if r.get("confidence") is not None:
                    confs.append(float(r["confidence"]))
                    weights.append(len(chunk))
        conf = float(np.average(confs, weights=weights)) if confs else None
        return {"text": " ".join(t for t in texts if t), "confidence": conf, "seconds": time.time() - t0,
                "chunks": len(texts), "mem_gb": mem}

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


def load(name: str) -> OmniASR:
    return OmniASR(name)


def unload(obj: OmniASR) -> None:
    obj.close()


# --------------------------------------------------------------------------- child
def _mem_gb(torch, device: str) -> float:
    """Resident memory of this process + memory the MPS driver holds for it (GB)."""
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except Exception:
        rss = 0
    mps = torch.mps.driver_allocated_memory() if device == "mps" else 0
    return (rss + mps) / 1e9


def _llm_confidence(pipe, inp, text: str, lang):
    """Geometric-mean token probability of `text` under the LLM (teacher forcing):
    exp(-mean token cross-entropy), the same loss the model was trained with."""
    import math
    import torch
    from fairseq2.datasets.batch import Seq2SeqBatch
    if not text.strip():
        return 0.0
    try:
        with torch.inference_mode():
            audio = list(pipe._build_audio_wavform_pipeline(inp).and_return())[0]
            tok = pipe.token_encoder(text)
            col = pipe.full_collater([{"audio_feature": audio, "text": tok}])
            batch = Seq2SeqBatch(source_seqs=col["audio_feature"]["seqs"].to(pipe.device, pipe.dtype),
                                 source_seq_lens=col["audio_feature"]["seq_lens"],
                                 target_seqs=col["text"]["seqs"].to(pipe.device),
                                 target_seq_lens=col["text"]["seq_lens"],
                                 example={"lang": [lang]} if lang else {})
            loss = float(pipe.model(batch))
        return round(math.exp(-loss), 4)
    except Exception as e:     # never break a transcription because scoring failed
        print("llm confidence failed: %s" % e, file=sys.stderr)
        return None


def _serve(card: str, device: str) -> int:
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
    from fairseq2.nn.batch_layout import BatchLayout
    out = sys.stdout
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    try:
        t0 = time.time()
        pipe = ASRInferencePipeline(model_card=card, device=device, dtype=dtype)
        out.write(json.dumps({"ready": True, "card": card, "device": device, "dtype": str(dtype),
                              "load_s": round(time.time() - t0, 1)}) + "\n")
        out.flush()
    except Exception as e:
        out.write(json.dumps({"ready": False, "error": "%s: %s" % (type(e).__name__, e)}) + "\n")
        out.flush()
        return 1
    is_ctc = isinstance(pipe.model, Wav2Vec2AsrModel)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            import soundfile as sf
            wav, sr = sf.read(req["wav"], dtype="float32")
            inp = [{"waveform": wav, "sample_rate": sr}]
            if is_ctc:
                with torch.inference_mode():
                    tensors = list(pipe._build_audio_wavform_pipeline(inp).and_return())
                    batch = pipe._create_batch_simple([(tensors[0], None)])
                    layout = BatchLayout(batch.source_seqs.shape, seq_lens=batch.source_seq_lens,
                                         device=batch.source_seqs.device)
                    logits, bl = pipe.model(batch.source_seqs, layout)
                    probs = torch.softmax(logits[0, : bl.seq_lens[0]].float(), dim=-1)
                    best_p, ids = probs.max(dim=-1)
                    keep = torch.ones_like(ids, dtype=torch.bool)
                    keep[1:] = ids[1:] != ids[:-1]
                    text = pipe.token_decoder(ids[keep])
                    blank = 0
                    emitted = ids != blank
                    conf = float(best_p[emitted].mean()) if bool(emitted.any()) else 0.0
                res = {"text": str(text), "confidence": round(conf, 4)}
            else:
                lang = req.get("lang")
                text = pipe.transcribe(inp, lang=[lang] if lang else None, batch_size=1)[0]
                res = {"text": str(text), "confidence": _llm_confidence(pipe, inp, str(text), lang)}
            res["mem_gb"] = round(_mem_gb(torch, device), 2)
            if device == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            res = {"error": "%s: %s" % (type(e).__name__, e)}
        out.write(json.dumps(res, ensure_ascii=False) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve"])
    ap.add_argument("--card", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    sys.exit(_serve(a.card, a.device))
