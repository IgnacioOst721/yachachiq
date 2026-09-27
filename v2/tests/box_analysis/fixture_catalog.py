"""A tiny fake museum catalog + a deterministic fake embedder (no model download) for the tests."""
from __future__ import annotations

import numpy as np

# culture -> (base colour of its thumbnails, embedding direction)
CULTURES = {
    "Moche": ((170, 60, 40), [1.0, 0.0, 0.0, 0.0]),
    "Nasca": ((200, 120, 50), [0.0, 1.0, 0.0, 0.0]),
    "Chimu": ((30, 30, 30), [0.0, 0.0, 1.0, 0.0]),
    "Maya": ((60, 140, 70), [0.0, 0.0, 0.0, 1.0]),
}
RAW = {"Moche": "Moche, north coast, Peru", "Nasca": "Nasca, south coast, Peru", "Chimu": "Chimú, north coast, Peru",
       "Maya": "Maya, Guatemala"}
MEDIUM = {"Moche": "Ceramic, slip", "Nasca": "Ceramic, polychrome slip", "Chimu": "Blackware ceramic", "Maya": "Jadeite"}
DIM = 16


def colour_embedding(img: np.ndarray) -> np.ndarray:
    """Fake embedder: nearest-culture blend from the mean colour, deterministic, unit norm."""
    px = np.asarray(img, np.float32).reshape(-1, 3)
    bg = (np.abs(px - [228, 228, 226]).max(axis=1) < 12) | (px.max(axis=1) < 8)   # query background / black box
    m = np.median(px[~bg] if (~bg).sum() > 10 else px, axis=0)      # median: robust to blended edge pixels
    w = []
    for _c, (col, _d) in CULTURES.items():
        w.append(np.exp(-np.sum((m - np.array(col)) ** 2) / (2 * 25.0 ** 2)))
    w = np.array(w) / (sum(w) + 1e-9)
    v = np.zeros(DIM)
    for wi, (_c, (_col, d)) in zip(w, CULTURES.items()):
        v[:4] += wi * np.array(d)
    v[4:] = 0.02 * np.sin(np.arange(DIM - 4))                      # same small offset for every image
    return v / np.linalg.norm(v)


class FakeEmbedder:
    dim = DIM

    def embed(self, images, batch: int = 32) -> np.ndarray:
        return np.stack([colour_embedding(np.asarray(im.convert("RGB") if hasattr(im, "convert") else im)) for im in images])


def build(root, per_culture: int = 12, seed: int = 0) -> list:
    """Write items.jsonl, thumbnails and an index (model key 'fake') under `root`."""
    from PIL import Image
    from yq.box.analysis.catalog import CatalogIndex, CatalogStore
    from yq.box.analysis.taxonomy import normalize_record
    rng = np.random.default_rng(seed)
    st = CatalogStore(root).ensure()
    ids, embs = [], []
    for c, (col, _d) in CULTURES.items():
        for k in range(per_culture):
            sid = "%s%d" % (c.lower(), k)
            img = np.clip(np.array(col) + rng.normal(0, 6, (48, 48, 3)), 0, 255).astype(np.uint8)
            rel = "thumbs/fake/%s.jpg" % sid
            (st.root / "thumbs" / "fake").mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(st.root / rel, quality=95)
            rec = {"id": "fake:" + sid, "source": "fake", "source_id": sid, "museum": "Museo de Prueba",
                   "title": "Vessel %s %d" % (c, k), "object_name": "Vessel", "culture": RAW[c], "period": "",
                   "date": "", "date_begin": 100 + k, "date_end": 600 + k, "medium": MEDIUM[c],
                   "classification": "Ceramics", "department": "", "geography": RAW[c], "dims_cm": {},
                   "url": "https://example.org/%s" % sid, "license": "CC0", "thumb": rel}
            st.append(normalize_record(rec))
            ids.append(rec["id"])
            embs.append(colour_embedding(np.asarray(Image.open(st.root / rel))))
    CatalogIndex.save("fake", np.array(embs), ids, {"model": "fake", "dim": DIM}, root)
    return ids


def query_image(colour, size: int = 96) -> tuple:
    """Object photo on a black background + its mask."""
    img = np.zeros((size, size, 3), np.uint8)
    mask = np.zeros((size, size), bool)
    mask[20:76, 30:66] = True
    img[mask] = colour
    return img, mask
