"""Image embeddings (SigLIP 2, Apache-2.0) for the catalog and for query photos.

Model loading goes through the Mac worker's model manager when it is running (routes_box
registers "box-siglip2"), otherwise a module-level cache is used (tools, tests). Heavy: load it
inside yq.common.heavylock when developing. Only the vision tower is loaded for images.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import settings

MODELS = {
    "siglip2-so400m-384": {"repo": "google/siglip2-so400m-patch14-384", "size_gb": 1.0, "dim": 1152},
    "siglip2-base-384": {"repo": "google/siglip2-base-patch16-384", "size_gb": 0.3, "dim": 768},
}
_cache: dict = {}
_lock = threading.Lock()


def device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, model_key: Optional[str] = None):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        self.key = model_key or settings.EMBED_MODEL
        spec = MODELS[self.key]
        self.device = device()
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        kw = {"local_files_only": not settings.ALLOW_DOWNLOAD}
        try:
            self.proc = AutoImageProcessor.from_pretrained(spec["repo"], local_files_only=True)
            self.model = SiglipVisionModel.from_pretrained(spec["repo"], dtype=dtype, local_files_only=True)
        except OSError:
            if kw["local_files_only"]:
                raise RuntimeError("modelo %s no está descargado y YQ_BOX_ALLOW_DOWNLOAD=0" % spec["repo"])
            self.proc = AutoImageProcessor.from_pretrained(spec["repo"])
            self.model = SiglipVisionModel.from_pretrained(spec["repo"], dtype=dtype)
        self.model.eval().to(self.device)
        self.dtype = dtype
        self.dim = spec["dim"]

    def embed(self, images: list, batch: int = 32) -> np.ndarray:
        """images: PIL images or RGB uint8 arrays -> (N, D) float32, L2-normalised."""
        import torch
        from PIL import Image
        out = []
        for s in range(0, len(images), batch):
            chunk = [Image.fromarray(im) if isinstance(im, np.ndarray) else im.convert("RGB") for im in images[s:s + batch]]
            px = self.proc(images=chunk, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            with torch.no_grad():
                f = self.model(pixel_values=px).pooler_output.float()
            f = torch.nn.functional.normalize(f, dim=-1)
            out.append(f.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def close(self) -> None:
        import gc
        del self.model
        gc.collect()
        try:
            import torch
            if self.device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass


def get_embedder(model_key: Optional[str] = None) -> Embedder:
    key = model_key or settings.EMBED_MODEL
    try:                                   # inside the Mac worker: share memory accounting with other domains
        from yq.macworker.modelmgr import models
        name = "box-" + key
        if name in models.registered():
            return models.get(name)
    except Exception:
        pass
    with _lock:
        if key not in _cache:
            _cache[key] = Embedder(key)
        return _cache[key]


def release(model_key: Optional[str] = None) -> None:
    with _lock:
        for k in list(_cache):
            if model_key is None or k == model_key:
                _cache.pop(k).close()


def _load_thumb(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def build_index(model_key: Optional[str] = None, root: Optional[Path] = None, chunk: int = 2000, batch: int = 32,
                log: Callable[[str], None] = print, limit: Optional[int] = None) -> int:
    """Embed every catalog thumbnail not yet in the index. Each chunk runs inside the heavy lock and
    the model is released between chunks so other engineers/jobs can use the Mac. Resumable."""
    from yq.common.heavylock import heavy
    from .catalog import CatalogIndex, CatalogStore
    key = model_key or settings.EMBED_MODEL
    store = CatalogStore(root)
    d = CatalogIndex.index_dir(key, root)
    ids, emb = [], np.zeros((0, MODELS[key]["dim"]), np.float16)
    if CatalogIndex.exists(key, root):
        import json
        ids = json.loads((d / "ids.json").read_text())
        emb = np.load(d / "emb.npy")
    have = set(ids)
    todo = [it for it in store.items() if it["id"] not in have and (store.root / it["thumb"]).exists()]
    if limit is not None:
        todo = todo[:limit]
    log("embed %s: %d in index, %d to embed" % (key, len(ids), len(todo)))
    t0 = time.time()
    done = 0
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        imgs, pid = [], []
        for it in part:
            try:
                imgs.append(_load_thumb(store.root / it["thumb"]))
                pid.append(it["id"])
            except Exception as e:
                log("skip %s: %s" % (it["id"], e))
        with heavy("box_analysis catalog embeddings"):
            e = get_embedder(key).embed(imgs, batch=batch)
            release(key)
        emb = np.concatenate([emb, e.astype(np.float16)])
        ids += pid
        CatalogIndex.save(key, emb, ids, {"model": key, "repo": MODELS[key]["repo"], "dim": int(emb.shape[1])}, root)
        done += len(pid)
        rate = done / max(1e-6, time.time() - t0)
        log("embed %s: %d/%d (%.1f img/s, ETA %.0f min)" % (key, done, len(todo), rate, (len(todo) - done) / max(rate, 1e-6) / 60))
    return len(ids)
