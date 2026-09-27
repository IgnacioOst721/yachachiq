"""Dimensions, volume, mass and density on synthetic scans rendered from known ground truth."""
from __future__ import annotations

import pytest

from yq.common.contracts import Measurement


def _m(result, name):
    return next(m for m in result.measurements if m["name"] == name)


@pytest.mark.parametrize("name", ["cylinder", "ellipsoid"])
def test_dimensions_within_1_5_percent(scans, analyzed, name):
    _folder, truth = scans[name]
    r = analyzed[name]
    for k in ("height", "width", "depth"):
        got, want = _m(r, k)["value"], truth["dims"][k]
        assert abs(got - want) / want < 0.015, (k, got, want)
        assert _m(r, k)["uncertainty"] > 0


@pytest.mark.parametrize("name", ["cylinder", "ellipsoid"])
def test_volume_within_3_percent(scans, analyzed, name):
    _folder, truth = scans[name]
    v = _m(analyzed[name], "volume_envelope")
    want = truth["volume_mm3"] / 1000.0
    assert abs(v["value"] - want) / want < 0.03, (v, want)
    # the stated 1-sigma uncertainty is honest: the true value is within 3 sigma
    assert abs(v["value"] - want) <= 3 * v["uncertainty"]


def test_mass_and_density(scans, analyzed):
    _folder, truth = scans["cylinder"]
    r = analyzed["cylinder"]
    assert _m(r, "mass")["value"] == pytest.approx(truth["mass_g"], abs=1.0)
    rho = _m(r, "density_apparent")
    assert rho["value"] == pytest.approx(truth["mass_g"] / (truth["volume_mm3"] / 1000.0), rel=0.04)
    assert "hueco" in rho["note"]
    assert any(f["title_es"] == "Densidad aparente" for f in r.findings)


def test_mass_rules():
    from yq.box.analysis.measure import density, mass_measurement
    w = []
    assert mass_measurement(None, w) is None and w
    m = mass_measurement({"grams": 100.0, "sigma_g": 0.1, "samples": [100.0, 100.2, 99.8], "stable": False}, [])
    assert m.uncertainty >= 2.0                      # unstable scale -> bigger error, never below 0.5 g
    d = density(Measurement("mass", 100, "g", 1), Measurement("volume_envelope", 50, "cm3", 1))
    assert d.value == pytest.approx(2.0) and 0 < d.uncertainty < 0.1


def test_artifacts_and_model(analyzed):
    r = analyzed["cylinder"]
    folder = r.folder
    from pathlib import Path
    for key in ("model_glb", "model_view0", "rti_ptm", "uv_overlay", "thermal_anomaly"):
        assert key in r.artifacts, (key, r.warnings)
        assert (Path(folder) / r.artifacts[key]).is_file()
    import trimesh
    m = trimesh.load(Path(folder) / r.artifacts["model_glb"], force="mesh")
    ext = m.bounds[1] - m.bounds[0]                 # glTF metres, Y up
    assert ext[1] == pytest.approx(0.100, abs=0.003) and ext[0] == pytest.approx(0.080, abs=0.003)
