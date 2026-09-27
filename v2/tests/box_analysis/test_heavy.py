"""Heavy tests: the real SigLIP 2 embedder (downloaded once) and the real catalog when it exists.

Run inside the memory lock:
    .venvs/box_analysis/bin/python -m yq.common.heavylock .venvs/box_analysis/bin/python -m pytest -m heavy tests/box_analysis
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.heavy


def test_real_embedder_properties():
    from yq.box.analysis.embed import MODELS, Embedder
    e = Embedder("siglip2-base-384")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (300, 200, 3), dtype=np.uint8)
    red = np.zeros((200, 200, 3), np.uint8)
    red[..., 0] = 200
    v = e.embed([a, a[::-1].copy(), red])
    assert v.shape == (3, MODELS["siglip2-base-384"]["dim"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-3)
    assert v[0] @ v[1] > v[0] @ v[2]                  # a flipped noise image is closer than a flat red one
    e.close()


@pytest.mark.skipif(not (Path(os.path.expanduser("~/yq-data/catalog")) / "index").exists(), reason="catalog not built")
def test_real_catalog_leave_one_out_smoke():
    from yq.box.analysis import evaluate
    from yq.box.analysis.catalog import CatalogIndex
    root = Path(os.path.expanduser("~/yq-data/catalog"))
    keys = [d.name for d in (root / "index").iterdir() if CatalogIndex.exists(d.name, root)]
    res = evaluate.leave_one_out(keys[0], root, n=200, save_calibration=False)
    assert res["summary"]["culture/none"]["top1"] > 0.3
