"""Offline museum reference catalog (CONTRACTS.md: config.CATALOG_DIR, Mac side).

Layout under CATALOG_DIR:
    items.jsonl                 one JSON record per object (append-only while building)
    thumbs/<source>/<id>.jpg    <=384 px thumbnail (longest side)
    state/                      resume files of the builder (see catalog_sources.py)
    index/<model_key>/          emb.npy (N x D float16, L2-normalized), ids.json, meta.json

Records only come from open-access sources (CC0 / public domain images); each keeps its
museum, url and license. Retrieval never invents references: every hit is a stored record.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from yq.common import config

THUMB_PX = 384


def catalog_dir(root: Optional[Path] = None) -> Path:
    return Path(root) if root else Path(config.CATALOG_DIR)


class CatalogStore:
    """Thread-safe append-only record store."""

    def __init__(self, root: Optional[Path] = None):
        self.root = catalog_dir(root)
        self.items_path = self.root / "items.jsonl"
        self.thumbs = self.root / "thumbs"
        self.state = self.root / "state"
        self._lock = threading.Lock()
        self._ids: Optional[set] = None

    def ensure(self) -> "CatalogStore":
        for d in (self.root, self.thumbs, self.state):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def iter_items(self) -> Iterable[dict]:
        if not self.items_path.exists():
            return
        with open(self.items_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue  # a line cut by a crash; ignored

    def items(self) -> list[dict]:
        seen, out = set(), []
        for it in self.iter_items():
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            out.append(it)
        return out

    def ids(self) -> set:
        if self._ids is None:
            self._ids = {it["id"] for it in self.iter_items()}
        return self._ids

    def has(self, item_id: str) -> bool:
        return item_id in self.ids()

    def append(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            ids = self.ids()
            if rec["id"] in ids:
                return
            with open(self.items_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            ids.add(rec["id"])

    def thumb_path(self, rec: dict) -> Path:
        return self.root / rec["thumb"]

    def count(self) -> int:
        return len(self.ids())


def save_thumb(data: bytes, dest: Path, max_px: int = THUMB_PX) -> tuple:
    """Decode an image, shrink to max_px on the longest side, save as JPEG. Returns (w, h)."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    im = im.convert("RGB")
    im.thumbnail((max_px, max_px), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    im.save(tmp, "JPEG", quality=85, optimize=True)
    tmp.replace(dest)
    return im.size


# --- Vector index ------------------------------------------------------------------------------
class CatalogIndex:
    """Embeddings of every catalog thumbnail for one model, searched by cosine similarity.

    150k x 1152 float16 = 350 MB; a matrix product on the CPU takes ~0.1 s, so no ANN
    library is needed (faiss would add a dependency for no measurable gain at this size).
    """

    def __init__(self, emb, ids: list, items: dict, meta: dict, root: Optional[Path] = None):
        import numpy as np
        self.root = catalog_dir(root)
        self.emb = emb                        # (N, D) float16, L2-normalized rows
        self.ids = list(ids)
        self.items = items                    # id -> record
        self.meta = meta
        self._row = {i: k for k, i in enumerate(self.ids)}
        self._np = np

    @staticmethod
    def index_dir(model_key: str, root: Optional[Path] = None) -> Path:
        return catalog_dir(root) / "index" / model_key

    @classmethod
    def exists(cls, model_key: str, root: Optional[Path] = None) -> bool:
        d = cls.index_dir(model_key, root)
        return (d / "emb.npy").exists() and (d / "ids.json").exists()

    @classmethod
    def available(cls, root: Optional[Path] = None) -> list:
        """Model keys that have a built index, best first (so400m > base)."""
        d = catalog_dir(root) / "index"
        keys = [p.name for p in d.iterdir() if p.is_dir() and cls.exists(p.name, root)] if d.is_dir() else []
        return sorted(keys, key=lambda k: (0 if "so400m" in k else 1, k))

    _cache: dict = {}

    @classmethod
    def load(cls, model_key: str, root: Optional[Path] = None, mmap: bool = True) -> "CatalogIndex":
        """Load (cached per process until the index or the records change). Records are re-normalized
        with the current taxonomy, so vocabulary fixes apply without rewriting items.jsonl."""
        d = cls.index_dir(model_key, root)
        st = CatalogStore(root)
        try:
            stamp = ((d / "emb.npy").stat().st_mtime, st.items_path.stat().st_size)
        except OSError:
            stamp = None
        key = (model_key, str(catalog_dir(root)))
        hit = cls._cache.get(key)
        if hit and hit[0] == stamp and stamp is not None:
            return hit[1]
        idx = cls._load(model_key, root, mmap)
        cls._cache[key] = (stamp, idx)
        return idx

    @classmethod
    def _load(cls, model_key: str, root: Optional[Path] = None, mmap: bool = True) -> "CatalogIndex":
        import numpy as np
        from .taxonomy import normalize_record
        d = cls.index_dir(model_key, root)
        if not cls.exists(model_key, root):
            raise FileNotFoundError("no catalog index for %s in %s (run tools/box_analysis_build_catalog.py embed)" % (model_key, d))
        emb = np.load(d / "emb.npy", mmap_mode="r" if mmap else None)
        ids = json.loads((d / "ids.json").read_text())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        store = CatalogStore(root)
        items = {it["id"]: normalize_record(it) for it in store.iter_items()}
        keep = [k for k, i in enumerate(ids) if i in items]
        if len(keep) != len(ids):
            emb = np.asarray(emb)[keep]
            ids = [ids[k] for k in keep]
        return cls(emb, ids, items, meta, root)

    @staticmethod
    def save(model_key: str, emb, ids: list, meta: dict, root: Optional[Path] = None) -> Path:
        import numpy as np
        d = CatalogIndex.index_dir(model_key, root)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "emb.tmp.npy"
        np.save(tmp, np.asarray(emb, dtype=np.float16))
        tmp.replace(d / "emb.npy")
        (d / "ids.json").write_text(json.dumps(list(ids)))
        meta = dict(meta, count=len(ids), saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        return d

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query, k: int = 50, exclude: Optional[set] = None) -> list:
        """query: (D,) or (Q, D) normalized embeddings. Multi-view queries are fused by taking,
        for each catalog object, the best similarity over the views plus a small bonus for the
        mean (objects similar from every side rank higher). Returns [(id, score), ...]."""
        np = self._np
        q = np.atleast_2d(np.asarray(query, dtype=np.float32))
        # errstate: NumPy 2.x + Apple Accelerate raises spurious divide/overflow warnings in large
        # matmuls on macOS; the result is finite and correct (checked on the 14k catalog).
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = np.asarray(self.emb, dtype=np.float32) @ q.T      # (N, Q)
        score = sims.max(axis=1) * 0.8 + sims.mean(axis=1) * 0.2 if q.shape[0] > 1 else sims[:, 0]
        if exclude:
            for i in exclude:
                r = self._row.get(i)
                if r is not None:
                    score[r] = -np.inf
        k = min(k, len(score))
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.argsort(-score[top])]
        return [(self.ids[r], float(score[r])) for r in top]

    def vector(self, item_id: str):
        return self._np.asarray(self.emb[self._row[item_id]], dtype=self._np.float32)


def public_reference(rec: dict, score: float, image: str = "") -> dict:
    """The `similar` entry shown to visitors (Identification.similar)."""
    return {"id": rec["id"], "title": rec.get("title", ""), "culture": rec.get("culture_norm") or rec.get("culture", ""),
            "culture_raw": rec.get("culture", ""), "date": rec.get("date", ""), "medium": rec.get("medium", ""),
            "museum": rec.get("museum", ""), "url": rec.get("url", ""), "license": rec.get("license", ""),
            "image": image, "score": round(float(score), 4)}
