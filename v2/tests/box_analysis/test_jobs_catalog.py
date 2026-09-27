"""Mac worker jobs end-to-end through the job manager; catalog records, store and taxonomy."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from tests.box_analysis import fixture_catalog as fx


def _zip_folder(folder) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in folder.rglob("*"):
            if p.is_file() and "analysis" not in p.parts:
                z.write(p, p.relative_to(folder.parent))
    return buf.getvalue()


def test_scan_analyze_and_identify_jobs(tmp_path, monkeypatch):
    from yq.box.analysis import embed, settings
    from yq.box.analysis.synthetic import Box
    from yq.box.analysis.synthetic_scan import make_scan
    from yq.common import config
    from yq.macworker import routes_box
    from yq.macworker.jobs import JobManager
    from yq.macworker.modelmgr import ModelManager
    fx.build(config.CATALOG_DIR)
    monkeypatch.setattr(settings, "EMBED_MODEL", "fake")
    monkeypatch.setattr(settings, "USE_VLM", False)
    monkeypatch.setattr(settings, "REFINE_SURFACE", False)
    monkeypatch.setattr(embed, "get_embedder", lambda key=None: fx.FakeEmbedder())
    folder = tmp_path / "scan-test"
    make_scan(folder, Box(70, 50, 60, yaw_deg=10), size=(388, 291), angles=range(0, 360, 30), rti=False, uv=False,
              thermal=False, calib_dir=None)
    jobs, models = JobManager(root=tmp_path / "jobs"), ModelManager(budget_gb=10)
    routes_box.setup(jobs, models)
    assert {"scan_analyze", "identify"} <= set(jobs.kinds())
    assert any(n.startswith("box-siglip2") for n in models.registered()) and not models.loaded()
    ctx = {"found_where": "Trujillo", "region_hint": "costa_norte"}
    job = jobs.run_inline("scan_analyze", {"profile": "standard", "analyses": ["weight", "photogrammetry", "identify"],
                                           "lang": "spa_Latn", "context": ctx}, [("scan.zip", _zip_folder(folder))])
    assert job.status == "done", job.error
    res = job.result
    out = job.folder / "out"
    names = {m["name"] for m in res["measurements"]}
    assert {"mass", "height", "width", "depth", "volume_envelope", "density_apparent"} <= names
    assert (out / res["artifacts"]["model_glb"]).is_file()
    ident = res["identification"]
    assert ident and ident["image_only"] is not None and ident["context_effect_es"]
    for s in ident["similar"]:
        assert (out / s["image"]).is_file()
    photo = next((folder / "photogrammetry").glob("camA_000.jpg")).read_bytes()
    j2 = jobs.run_inline("identify", {"measurements": res["measurements"], "notes": ""}, [("a.jpg", photo)])
    assert j2.status == "done" and j2.result["similar"], j2.error


def test_source_records_and_taxonomy():
    from yq.box.analysis.catalog_sources import Builder
    from yq.box.analysis.taxonomy import excluded_classification, normalize_record
    met = {"objectID": 313256, "isPublicDomain": True, "primaryImageSmall": "https://x/y.jpg", "title": "Mirror-bearer",
           "objectName": "Male figure", "culture": "Maya", "period": "", "dynasty": "", "objectDate": "410–650 CE",
           "objectBeginDate": 410, "objectEndDate": 650, "medium": "Cordia wood (bocote), red hematite",
           "classification": "Wood-Sculpture", "department": "The Michael C. Rockefeller Wing", "country": "Guatemala or Mexico",
           "measurements": [{"elementName": "Overall", "elementMeasurements": {"Height": 35.8775, "Width": 22.86}}],
           "objectURL": "https://www.metmuseum.org/art/collection/search/313256", "accessionNumber": "1979.206.1063"}
    r = normalize_record(Builder.met_record(None, met))
    assert (r["culture_norm"], r["material_norm"], r["type_norm"], r["region_norm"]) == ("Maya", "wood", "figure", "Mesoamerica")
    assert normalize_record({"source": "cma", "title": "Figural Pendant", "object_name": "Sculpture"})["type_norm"] == "jewelry"
    assert r["dims_cm"] == {"h": 35.88, "w": 22.86} and r["license"].startswith("CC0")
    assert Builder.met_record(None, dict(met, isPublicDomain=False)) is None
    cma = {"id": 129799, "share_license_status": "CC0", "title": "Backrest of a Litter", "creation_date": "1185–1275",
           "creation_date_earliest": 1185, "creation_date_latest": 1275, "type": "Sculpture",
           "culture": ["Central Andes, North Coast, Chimú people, late Intermediate period"],
           "technique": "mixed media:  wood, gold alloy, pigment, shell inlay", "url": "https://clevelandart.org/art/1952.233",
           "images": {"web": {"url": "https://x/w.jpg"}}, "dimensions": {"overall": {"height": 0.604, "width": 0.95}}}
    c = normalize_record(Builder.cma_record(None, cma))
    assert c["culture_norm"] == "Chimu" and c["dims_cm"]["h"] == pytest.approx(60.4)
    assert excluded_classification("Prints") and not excluded_classification("Ceramics-Containers")
    n = normalize_record({"title": "Stirrup Spout Bottle", "culture": "Moche", "medium": "Ceramic, pigment"})
    assert (n["type_norm"], n["material_cls"]) == ("stirrup-spout vessel", "ceramic")


def test_store_resume_and_index_search(tmp_path):
    from yq.box.analysis.catalog import CatalogIndex, CatalogStore
    ids = fx.build(tmp_path)
    st = CatalogStore(tmp_path)
    st.append({"id": ids[0], "thumb": "x"})                     # duplicate ignored
    assert st.count() == len(ids) and len(st.items()) == len(ids)
    idx = CatalogIndex.load("fake", tmp_path)
    hits = idx.search(idx.vector(ids[0]), k=5, exclude={ids[0]})
    assert ids[0] not in [h for h, _s in hits] and len(hits) == 5
    assert all(h.split(":")[1][:4] == ids[0].split(":")[1][:4] for h, _s in hits)
