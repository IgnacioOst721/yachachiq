"""Closed-loop heating: the guard logic on synthetic frames, and the whole
thermal stage on the simulator + mock thermal camera."""
import json

import numpy as np
import pytest

from yq.box import layout
from yq.box import settings as S
from yq.box.heatguard import HeatGuard
from yq.common import config
from yq.common.contracts import ScanRequest, new_id


def _frames(rise_blob=0.0, rise_all=0.0, hot_corner=False, seed=0):
    rng = np.random.default_rng(seed)
    f = np.full((120, 160), 22.0, np.float32) + rng.normal(0, 0.04, (120, 160)).astype(np.float32)
    f[40:90, 50:110] += rise_all                     # the object
    f[55:70, 70:85] += rise_blob                     # a spot that heats faster (defect)
    if hot_corner:
        f[12:20, 60:70] = 80.0                       # lamp fixture already hot, inside the ROI
    return f


def _guard(**kw):
    g = HeatGuard(**kw)
    for i in range(20):
        g.add_baseline(_frames(seed=i))
    g.start()
    return g


def test_guard_stops_on_delta_t_from_the_hottest_spot():
    g = _guard(max_dt_c=2.0, max_abs_c=35.0)
    reason = None
    for k in range(1, 60):                           # object +0.05 C/step, defect spot +0.12 C/step
        reason = g.update(k * 0.115, _frames(rise_all=0.05 * k, rise_blob=0.07 * k, seed=100 + k))
        if reason:
            break
    assert reason == "delta_t"
    assert 2.0 <= g.max_dt_heating < 2.3              # stopped within one frame of the limit
    assert g.last["region"] == "warmed"
    assert k < 20                                    # the defect, not the average, decided


def test_guard_stops_on_absolute_temperature():
    g = _guard(max_dt_c=50.0, max_abs_c=24.0)
    reasons = [g.update(k * 0.115, _frames(rise_all=0.1 * k, seed=k)) for k in range(1, 40)]
    first = next(i for i, r in enumerate(reasons) if r)
    assert reasons[first] == "abs_limit" and 18 <= first <= 20      # 22 C + 0.1 C/frame reaches 24 C


def test_guard_ignores_things_already_hot_before_heating():
    g = HeatGuard(max_dt_c=5.0, max_abs_c=35.0)
    for i in range(10):
        g.add_baseline(_frames(seed=i, hot_corner=True))
    g.start()
    assert g.excluded_hot == 80                      # 8 x 10 hot pixels ignored
    assert g.update(0.1, _frames(seed=50, hot_corner=True, rise_all=1.0)) is None


def test_guard_stops_when_the_camera_freezes():
    g = _guard(max_frozen_s=1.0)
    frame = _frames(seed=7, rise_all=0.5)
    assert g.update(0.0, frame) is None
    t, reason = 0.0, None
    while reason is None and t < 3:
        t += 0.115
        reason = g.update(t, frame.copy())            # identical frames: FFC or a stalled stream
    assert reason == "camera_frozen" and 1.0 <= t < 1.3


def _thermal_scan(monkeypatch, **settings):
    from yq.box import scan
    for k, v in settings.items():
        monkeypatch.setattr(S, k, v)
    monkeypatch.setitem(S.PROFILES, "quick", {**S.PROFILES["quick"], "thermal_cool_s": 20.0})
    req = ScanRequest(scan_id=new_id("scan"), profile="quick", analyses=["weight", "thermal"])
    res = scan.run_scan(req)
    folder = config.SCANS_DIR / res.scan_id
    meta = json.loads((folder / "thermal" / "meta.json").read_text()) if (folder / "thermal").is_dir() else None
    return res, folder, meta


def test_mock_scan_stops_early_with_a_low_delta_t(mock_all, monkeypatch):
    res, folder, tm = _thermal_scan(monkeypatch, THERMAL_MAX_DT_C=1.5)
    assert res.ok and layout.validate_scan_folder(folder) == []
    assert tm["stop_reason"] == "delta_t"
    assert tm["heat_s"] < 0.5 * tm["heat_planned_s"]
    assert tm["heat_off_s"] - tm["heat_on_s"] == pytest.approx(tm["heat_s"], abs=1e-3)
    assert 1.5 <= tm["max_dt_c"] < 2.2                # tiny overshoot only
    assert tm["limits"] == {"max_dt_c": 1.5, "max_abs_c": 35.0}
    times = np.load(folder / "thermal" / "times.npy")
    assert times[-1] == pytest.approx(tm["heat_off_s"] + 20.0, abs=0.3)   # cooling counted from switch-off
    assert any("detenido antes" in w for w in res.warnings)


def test_mock_scan_stops_on_absolute_limit(mock_all, monkeypatch):
    res, folder, tm = _thermal_scan(monkeypatch, THERMAL_MAX_ABS_C=23.0)
    assert res.ok and tm["stop_reason"] == "abs_limit"
    assert 23.0 <= tm["max_temp_c"] < 23.6 and tm["heat_s"] < tm["heat_planned_s"]


def test_mock_scan_default_limits_are_recorded(mock_all, monkeypatch):
    res, folder, tm = _thermal_scan(monkeypatch)
    assert tm["stop_reason"] in ("time", "delta_t")
    assert tm["max_dt_c"] < S.THERMAL_MAX_DT_C + 0.7 and tm["max_temp_c"] < S.THERMAL_MAX_ABS_C
    assert tm["heating_enabled"] is True and tm["heat_s"] <= 15.3


def test_heating_disabled_means_no_thermography(mock_all, monkeypatch):
    res, folder, tm = _thermal_scan(monkeypatch, THERMAL_HEAT_ENABLED=False)
    meta = json.loads((folder / "meta.json").read_text())
    assert tm is None and "thermal" not in meta["analyses"] and "thermal" not in meta["captured"]
    assert res.ok and layout.validate_scan_folder(folder) == []
    assert any("desactivado" in w for w in res.warnings)
    from yq.box.device import get_box
    assert not [e for e in get_box().sim.sent_events if e.get("ch") == "halogen"]   # never switched on


def test_cli_no_heat_removes_thermal(mock_all, monkeypatch):
    from yq.box import cli, scan
    seen = {}
    monkeypatch.setattr(scan, "preflight", lambda: {"ok": True, "problems_es": [], "weight_g": 800})

    def fake_run(req, on_progress=None):
        seen["analyses"] = req.analyses
        from yq.common.contracts import ScanResult
        return ScanResult(scan_id=req.scan_id, folder="x", profile=req.profile, started=0)

    monkeypatch.setattr(scan, "run_scan", fake_run)
    cli.cmd_scan(cli.build_parser().parse_args(["scan", "--profile", "quick", "--no-heat"]))
    assert "thermal" not in seen["analyses"] and "photogrammetry" in seen["analyses"]
