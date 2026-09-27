"""Identification with a tiny fixture catalog, a fake embedder and a mocked VLM; visitor context rules."""
from __future__ import annotations

import numpy as np
import pytest

from tests.box_analysis import fixture_catalog as fx


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    from yq.box.analysis import embed, settings
    from yq.common import config
    root = config.CATALOG_DIR
    ids = fx.build(root)
    monkeypatch.setattr(settings, "EMBED_MODEL", "fake")
    monkeypatch.setattr(settings, "USE_VLM", False)
    monkeypatch.setattr(embed, "get_embedder", lambda key=None: fx.FakeEmbedder())
    return set(ids)


def _run(colour, context=None, out=None, **kw):
    from yq.box.analysis.identify import identify
    img, mask = fx.query_image(colour)
    return identify([img], [], context=context, out_dir=out, masks=[mask], **kw)


def test_retrieval_only_real_references(catalog, tmp_path):
    r = _run((172, 62, 42), out=tmp_path)
    assert r["culture"] == "Moche" and r["material"] and r["period"]
    assert r["similar"] and all(s["id"] in catalog for s in r["similar"])       # never invented
    assert all((tmp_path / s["image"]).is_file() for s in r["similar"])
    assert r["image_only"]["culture"] == "Moche" and r["context_effect_es"] == ""
    assert 0 < r["confidence"] <= 1 and "knn" in r["engine"]


class _Fixed(fx.FakeEmbedder):
    """Returns the exact mid-point of the Moche and Nasca clusters: a perfect tie between them."""
    def embed(self, images, batch=32):
        a = fx.colour_embedding(np.full((8, 8, 3), fx.CULTURES["Moche"][0], np.uint8))
        b = fx.colour_embedding(np.full((8, 8, 3), fx.CULTURES["Nasca"][0], np.uint8))
        v = (a + b) / np.linalg.norm(a + b)
        return np.stack([v] * len(images))


def test_context_breaks_ties_but_cannot_create(catalog, monkeypatch):
    from yq.box.analysis import embed
    monkeypatch.setattr(embed, "get_embedder", lambda key=None: _Fixed())
    mid = (185, 90, 45)
    base = _run(mid)
    south = _run(mid, {"region_hint": "costa_sur"})
    north = _run(mid, {"found_where": "cerca de Chan Chan, Trujillo"})
    assert {south["culture"], north["culture"]} == {"Nasca", "Moche"}
    assert south["image_only"]["culture"] == base["culture"] == north["image_only"]["culture"]
    changed = south if south["culture"] != base["culture"] else north
    assert changed["context_effect_es"].startswith("El lugar ayudó a decidir")
    monkeypatch.setattr(embed, "get_embedder", lambda key=None: fx.FakeEmbedder())
    maya = _run((60, 140, 70), {"region_hint": "costa_norte"})   # clearly Maya: the place cannot make it Moche
    assert maya["culture"] == "Maya"
    assert maya["context_effect_es"] == "El lugar no coincide con lo que se ve; se priorizó la imagen."


def test_vlm_breaks_near_tie_and_is_validated(catalog, monkeypatch):
    from yq.box.analysis import embed, identify as idm, settings
    monkeypatch.setattr(embed, "get_embedder", lambda key=None: _Fixed())
    monkeypatch.setattr(settings, "USE_VLM", True)
    monkeypatch.setattr(idm, "ask_vlm", lambda images, prompt: None)
    mid = (185, 90, 45)
    base = _run(mid)
    other = "Nasca" if base["culture"] == "Moche" else "Moche"
    reply = {"object_type": "stirrup-spout bottle", "material": "ceramic", "culture": other, "period": "100-600 CE",
             "region": "Peru", "confidence": 0.7, "evidence_es": ["forma"], "description_es": "Una vasija."}
    monkeypatch.setattr(idm, "ask_vlm", lambda images, prompt: dict(reply))
    r = _run(mid)
    assert r["culture"] == other and "+vlm" in r["engine"] and r["description_es"] == "Una vasija."
    monkeypatch.setattr(idm, "ask_vlm", lambda images, prompt: dict(reply, culture="Egyptian"))
    r2 = _run(mid)
    assert r2["culture"] == base["culture"] and r2["confidence"] < base["confidence"]   # unsupported claim ignored


def test_vlm_contract_call(catalog, monkeypatch):
    from yq.box.analysis import identify as idm, settings
    from yq.macworker.models import vlm
    seen = {}

    def fake_ask(images, prompt, max_tokens=600):
        seen["prompt"], seen["n"] = prompt, len(images)
        return 'Sure: {"object_type": "bowl", "material": "ceramic", "culture": "Moche", "period": "", "region": "", ' \
               '"confidence": 0.5, "evidence_es": [], "description_es": "x"}'
    monkeypatch.setattr(settings, "USE_VLM", True)
    monkeypatch.setattr(vlm, "ask", fake_ask)
    r = _run((172, 62, 42), {"found_where": "Trujillo"})
    assert "+vlm" in r["engine"] and seen["n"] == 1
    assert "puede ser incorrecto" in seen["prompt"] and "Museo de Prueba" in seen["prompt"]


def test_mock_and_missing_catalog(monkeypatch, tmp_path):
    from yq.box.analysis.identify import identify
    img, mask = fx.query_image((172, 62, 42))
    assert identify([img], masks=[mask], model_key="nope")["engine"] == "unavailable"
    monkeypatch.setenv("YQ_MOCK_BOX_ANALYSIS", "1")
    assert identify([img], masks=[mask])["engine"] == "mock"


def test_parse_context_and_bounded_prior():
    from yq.box.analysis.context import apply_prior, parse_context
    assert parse_context({"found_where": "lo encontró mi tío en Nasca"}, use_llm=False)["zone"] == "costa_sur"
    assert parse_context({"region_hint": "altiplano"})["confidence"] == 1.0
    assert parse_context({"found_where": "no me acuerdo"}, use_llm=False)["confidence"] == 0.0
    h = parse_context({"region_hint": "costa_norte"})
    scores = {"Moche": 0.40, "Nasca": 0.45, "Maya": 0.05, "Chimu": 0.10}
    regions = {"Moche": "Andes", "Nasca": "Andes", "Maya": "Mesoamerica", "Chimu": "Andes"}
    new, mult = apply_prior(scores, regions, h, max_boost=1.5)
    assert mult["Moche"] == pytest.approx(1.5) and mult["Chimu"] == pytest.approx(1.0)   # below support: no boost
    assert max(new, key=new.get) == "Moche" and mult["Maya"] < 1
    assert all(1 / 1.5 <= m <= 1.5 for m in mult.values())
