"""RTI normals on a relief tile with known normals, PTM format, UV zones, thermography defects."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="module")
def tile_scan(tmp_path_factory):
    from yq.box.analysis.geometry import nominal_leds
    from yq.box.analysis.synthetic_relief import ReliefTile, write_rti_scan
    from yq.box.analysis.synthetic_scan import true_cameras, write_calibration
    root = tmp_path_factory.mktemp("tile")
    cams = true_cameras((776, 582), distortion=False)
    write_calibration(cams, root / "calib", nominal_leds())
    tile = ReliefTile()
    truth = write_rti_scan(root / "scan", tile, cams["B"], nominal_leds())
    return root, tile, truth


def _angular_errors(res, truth, mesh_prior=True):
    info = res["info"]
    x0, y0 = info["_crop"]
    mask = info["_mask"]
    h, w = mask.shape
    nt = truth["normals"][y0:y0 + h, x0:x0 + w]
    hit = truth["hit"][y0:y0 + h, x0:x0 + w]
    nw = np.full(mask.shape + (3,), np.nan)
    nw[mask] = info["_normals_world"]
    inner = cv2.erode(hit.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    sel = mask & inner & (nt[..., 2] > 0.5) & np.isfinite(nw[..., 0])
    return np.degrees(np.arccos(np.clip(np.sum(nw[sel] * nt[sel], 1), -1, 1))), sel, nt


def test_rti_normals_and_web_ptm(tile_scan, tmp_path):
    from yq.box.analysis.calib import BoxCalibration
    from yq.box.analysis.rti_run import analyze_rti
    from yq.box.analysis.scanio import load_scan
    root, tile, truth = tile_scan
    scan = load_scan(root / "scan")
    cal = BoxCalibration.load(scan.meta, root / "calib")
    V, F = tile.mesh(0.5)
    res = analyze_rti(scan, cal, tmp_path, V, F)
    ang, sel, nt = _angular_errors(res, truth)
    assert sel.sum() > 5000
    assert ang.mean() < 3.0 and np.median(ang) < 1.5, (ang.mean(), np.median(ang))
    relief = (nt[..., 2] < 0.995)[sel]                       # incisions and the boss only
    assert relief.sum() > 500 and ang[relief].mean() < 8.0, ang[relief].mean()
    meta = json.loads((tmp_path / res["artifacts"]["rti_ptm"]).read_text())
    assert meta["format"] == "yq-ptm-1" and len(meta["scale"]) == 6 and len(meta["bias"]) == 6
    for f in meta["files"].values():
        assert (tmp_path / f).is_file()
    c012 = cv2.imread(str(tmp_path / meta["files"]["coef012"]))
    assert c012.shape[:2] == (meta["height"], meta["width"])
    assert res["findings"] and res["findings"][0]["analysis"] == "rti"


def test_ptm_fit_recovers_biquadratic():
    from yq.box.analysis.rti import eval_ptm, fit_ptm
    rng = np.random.default_rng(0)
    lu = rng.uniform(-0.9, 0.9, (12, 50))
    lv = rng.uniform(-0.9, 0.9, (12, 50))
    c = rng.normal(0, 1, (50, 6))
    Y = eval_ptm(c[None], lu, lv)
    got = fit_ptm(Y, lu, lv, ridge=1e-9)
    assert np.allclose(got, c, atol=1e-4)


def test_uv_zone_segmented(scans, analyzed):
    from yq.box.analysis.calib import BoxCalibration
    from yq.box.analysis.scanio import load_scan
    from yq.box.analysis.uv import analyze_uv
    folder, truth = scans["cylinder"]
    scan = load_scan(folder)
    res = analyze_uv(scan, BoxCalibration.load(scan.meta), folder / "analysis")
    zones = res["info"]["_seg"]["zones"]
    assert len(zones) >= 1
    tm = np.array(truth["uv_zone"]["_mask"], bool)
    best = max(zones, key=lambda z: (z["_mask"] & tm).sum())
    iou = (best["_mask"] & tm).sum() / float((best["_mask"] | tm).sum())
    assert iou > 0.9, iou            # one physical zone comes out as ONE zone (was split in 4, IoU 0.50)
    # the object's shaded side (same hue, darker) is lighting, not a zone: nothing else is reported
    assert len(zones) == 1, [(z["area_px"], round(z["delta_e"], 1)) for z in zones]
    assert "especialista" in res["findings"][0]["detail_es"]


@pytest.mark.parametrize("depth,expect", [(1.0, True), (60.0, False)])
def test_thermal_defect_detected_in_place(depth, expect):
    from yq.box.analysis import thermo
    from yq.box.analysis.geometry import nominal_camera
    from yq.box.analysis.synthetic import Cylinder
    from yq.box.analysis.synthetic_physics import simulate_thermal
    seq, times, meta, truth = simulate_thermal(Cylinder(45, 110), nominal_camera("T"), defect={"depth_mm": depth})
    res = thermo.anomaly_maps(seq, times, meta)
    assert res["clean"]["dropped_frames"] > 0 and res["clean"]["offset_jumps"]
    an, _score = thermo.detect_anomalies(res["maps"], res["mask"])
    if expect:
        assert an and np.hypot(an[0]["u"] - truth["u"], an[0]["v"] - truth["v"]) < 3.0
        assert an[0]["warmer_late"] and an[0]["confidence"] >= 0.5
    else:
        assert not an


def test_rti_reads_the_capture_list_format(scans, tmp_path):
    """BOX-CAPTURE writes rti/lights.json as a plain list of the 8 lights (CONTRACTS §4); the analysis
    used to expect {"camera","platter_deg","lights"} and crashed ('list' object has no attribute 'get')."""
    import json
    import shutil
    from yq.box.analysis.calib import BoxCalibration
    from yq.box.analysis.rti_run import analyze_rti
    from yq.box.analysis.scanio import load_scan
    folder, _truth = scans["cylinder"]
    copy = tmp_path / "scan-list-lights"
    shutil.copytree(folder, copy)
    f = copy / "rti" / "lights.json"
    d = json.loads(f.read_text())
    f.write_text(json.dumps(d["lights"] if isinstance(d, dict) else d))
    scan = load_scan(copy)
    res = analyze_rti(scan, BoxCalibration.load(scan.meta), copy / "analysis")
    assert res and res.get("artifacts"), res


def test_identify_uses_the_catalog_that_exists(tmp_path, monkeypatch):
    """The default embedding model had no index (catalog built with siglip2-base) -> 'unavailable'."""
    from yq.box.analysis import identify as I
    from yq.box.analysis import settings as S
    from yq.box.analysis.catalog import CatalogIndex
    d = tmp_path / "catalog" / "index" / "siglip2-base-384"
    d.mkdir(parents=True)
    (d / "emb.npy").write_bytes(b"")
    (d / "ids.json").write_text("[]")
    assert CatalogIndex.available(tmp_path / "catalog") == ["siglip2-base-384"]
    monkeypatch.setattr(S, "EMBED_MODEL", "siglip2-so400m-384")
    seen = {}
    monkeypatch.setattr(CatalogIndex, "load", classmethod(lambda cls, key, root=None, mmap=True:
                                                          seen.setdefault("key", key) and (_ for _ in ()).throw(RuntimeError("stop"))))
    try:
        I.identify([np.zeros((8, 8, 3), np.uint8)], catalog_root=tmp_path / "catalog")
    except RuntimeError:
        pass
    assert seen["key"] == "siglip2-base-384"


def test_uv_curvature_bands_are_not_zones():
    """A vessel's shoulder/rim/base catch 1.5-2x more UV (same colour): that is lighting, not a
    restoration. Only the real zone (different colour, much brighter) must be reported."""
    from yq.box.analysis.uv import segment_zones
    h, w = 240, 200
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx - 100) / 80.0) ** 2 + ((yy - 120) / 110.0) ** 2 <= 1.0
    base = np.array([0.20, 0.22, 0.30])                       # bluish ceramic fluorescence
    F = np.zeros((h, w, 3)) + base
    F[(yy > 40) & (yy < 60)] *= 1.9                           # shoulder band: same hue, 1.9x
    F[(yy > 200) & (yy < 215)] *= 1.7                         # base edge: same hue, 1.7x
    F[(xx > 30) & (xx < 60) & (yy > 100) & (yy < 150)] *= 1.6  # side patch facing the lamp
    real = (xx > 120) & (xx < 160) & (yy > 90) & (yy < 140)
    F[real] = [1.6, 1.9, 0.9]                                 # restoration: yellow-green, much brighter
    F = F * mask[..., None]
    seg = segment_zones(F.astype(np.float32), mask)
    assert len(seg["zones"]) == 1, [(z["area_px"], round(z["delta_e"], 1), round(z["brightness_ratio"], 2))
                                    for z in seg["zones"]]
    z = seg["zones"][0]["_mask"]
    iou = (z & real).sum() / float((z | real).sum())
    assert iou > 0.8, iou
