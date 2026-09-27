"""Full scan in mock mode (simulator + synthetic cameras + mock thermal), the §4
layout validator, cancel and door handling."""
import json
import shutil
import threading

import numpy as np
import pytest

from yq.box import layout
from yq.common import config
from yq.common.contracts import ScanRequest, new_id


def _quick_scan():
    from yq.box import scan
    progress = []
    req = ScanRequest(scan_id=new_id("scan"), profile="quick")
    res = scan.run_scan(req, progress.append)
    return res, progress


def test_full_mock_scan_produces_a_valid_folder(mock_all, tmp_path):
    res, progress = _quick_scan()
    assert res.ok, res.warnings
    assert layout.validate_scan_folder(res.folder, require_result=True) == []
    folder = config.SCANS_DIR / res.scan_id
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["captured"] == ["weight", "uv", "rti", "photogrammetry", "thermal"]
    assert meta["cameras"]["A"]["focus"] == 300 and meta["box"]["simulated"]
    poses = json.loads((folder / "photogrammetry" / "poses.json").read_text())
    assert len(poses) == 36                                    # 18 stops x 2 cameras
    for name, p in poses.items():
        assert p["platter_deg"] == pytest.approx(p["nominal_deg"], abs=0.01)
    w = json.loads((folder / "weight.json").read_text())
    assert w["grams"] == pytest.approx(812.5, abs=1.0) and w["stable"]
    tm = json.loads((folder / "thermal" / "meta.json").read_text())
    assert 0 < tm["heat_off_s"] - tm["heat_on_s"] <= 15.3            # closed loop may stop earlier
    assert tm["stop_reason"] in ("time", "delta_t") and tm["max_dt_c"] < 5.7
    seq = np.load(folder / "thermal" / "sequence.npy")
    assert seq.shape[1:] == (120, 160) and seq.dtype == np.float32
    assert seq.max() > tm["ambient_c"] + 2                     # the halogen really heated the object
    result = json.loads((folder / "result.json").read_text())
    assert result["measurements"][0]["name"] == "mass"
    fr = [p.fraction for p in progress]
    assert fr == sorted(fr) and fr[-1] == 1.0
    assert all(p.message_es for p in progress)
    _everything_off_after(res)
    _validator_reports_broken_folders(res, tmp_path)
    _package_downscales_photos(res, tmp_path)


def _everything_off_after(res):
    from yq.box.device import get_box
    st = get_box().status()
    on = {k for k, v in st["lights"].items() if v["level"] > 0}
    assert on <= {"fan"} and not st["motor"]["moving"]       # only the cooling fan may run


def _validator_reports_broken_folders(res, tmp_path):
    bad = tmp_path / res.scan_id
    shutil.copytree(res.folder, bad)
    (bad / "rti" / "led3.jpg").unlink()
    (bad / "photogrammetry" / "camB_040.jpg").write_bytes(b"not a jpeg")
    np.save(bad / "thermal" / "sequence.npy", np.zeros((5, 60, 80), np.float64))
    meta = json.loads((bad / "meta.json").read_text())
    del meta["calibration"]
    (bad / "meta.json").write_text(json.dumps(meta))
    probs = layout.validate_scan_folder(bad)
    text = "\n".join(probs)
    for needle in ("rti/led3.jpg", "camB_040.jpg no es un JPEG", "sequence.npy", "calibration"):
        assert needle in text, (needle, probs)
    assert layout.validate_scan_folder(tmp_path / "nope") == ["la carpeta %s no existe" % (tmp_path / "nope")]


def test_preflight(mock_all):
    from yq.box import scan
    from yq.box.device import get_box
    pf = scan.preflight()
    assert pf["ok"] and pf["weight_g"] == pytest.approx(812.5, abs=1.5)
    box = get_box()
    box.sim.set_door("front", False)
    box.sim.place_object(0.0)
    pf = scan.preflight()
    assert not pf["ok"]
    assert any("puertas" in p for p in pf["problems_es"]) and any("objeto" in p for p in pf["problems_es"])


def test_cancel_turns_everything_off(mock_all):
    from yq.box import scan
    from yq.box.device import get_box
    cancel = threading.Event()

    def on_progress(p):
        if p.stage == "photogrammetry" and p.detail.get("index") == 3:
            cancel.set()

    res = scan.run_scan(ScanRequest(scan_id=new_id("scan"), profile="quick"), on_progress, cancel)
    assert not res.ok and any("cancelado" in w for w in res.warnings)
    st = get_box().status()
    assert all(v["level"] == 0 for v in st["lights"].values()) and not st["motor"]["moving"]
    assert json.loads((config.SCANS_DIR / res.scan_id / "result.json").read_text())["ok"] is False


def test_door_opened_mid_scan_pauses_until_closed(mock_all):
    from yq.box import scan
    from yq.box.device import get_box
    box = get_box()
    state = {"opened": False}

    def on_progress(p):
        if p.stage == "rti" and not state["opened"]:
            state["opened"] = True
            box.sim.set_door("front", False)
            threading.Timer(0.4, box.sim.set_door, ("front", True)).start()

    res = scan.run_scan(ScanRequest(scan_id=new_id("scan"), profile="quick", analyses=["weight", "uv", "rti"]),
                        on_progress)
    assert res.ok, res.warnings
    assert layout.validate_scan_folder(res.folder) == []


def test_mac_result_is_merged(mock_all, monkeypatch):
    """With a (fake) Mac answering, artifacts land in analysis/ and the result is merged."""
    from yq.box import package, scan
    monkeypatch.setattr(package, "analyze_on_mac", lambda folder, profile, analyses, on_progress=None, context=None: {
        "scan_id": "x", "folder": "x", "profile": profile, "started": 0, "measurements":
        [{"name": "height", "value": 118.0, "unit": "mm", "uncertainty": 1.0}], "artifacts":
        {"mesh": "analysis/mesh.glb"}, "findings": [], "warnings": ["mac: ok"], "ok": True})
    res = scan.run_scan(ScanRequest(scan_id=new_id("scan"), profile="quick", analyses=["weight"]))
    names = [m["name"] for m in res.measurements]
    assert names == ["mass", "height"] and res.artifacts == {"mesh": "analysis/mesh.glb"} and res.ok
    assert "mac: ok" in res.warnings


def _package_downscales_photos(res, tmp_path):
    import zipfile
    import cv2
    from yq.box import package
    z = package.package_scan(res.folder, tmp_path / "scan.zip", long_side=160)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        img = cv2.imdecode(np.frombuffer(zf.read("photogrammetry/camA_000.jpg"), np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 160
    assert "thermal/sequence.npy" in names and "result.json" not in names and "meta.json" in names
