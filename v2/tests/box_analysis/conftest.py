"""Shared synthetic scans for the BOX-ANALYSIS tests (rendered once per test session)."""
from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore", category=RuntimeWarning)

SIZE = (776, 582)          # 1/6 of the IMX519 resolution: enough for sub-mm accuracy, fast to render


@pytest.fixture(scope="session")
def scans(tmp_path_factory):
    """{name: (folder, truth)} for a cylinder and an ellipsoid with every analysis captured."""
    from yq.box.analysis.synthetic import Cylinder, Ellipsoid
    from yq.box.analysis.synthetic_scan import make_scan
    root = tmp_path_factory.mktemp("synthetic_scans")
    out = {}
    for name, shape, kw in (("cylinder", Cylinder(40.0, 100.0), {}),
                            ("ellipsoid", Ellipsoid(50.0, 35.0, 60.0), {"rti": False, "uv": False, "thermal": False})):
        folder = root / ("scan-%s" % name)
        truth = make_scan(folder, shape, size=SIZE, angles=range(0, 360, 15), calib_dir=root / ("calib-%s" % name), **kw)
        out[name] = (folder, truth)
    return out


@pytest.fixture(scope="session")
def analyzed(scans, tmp_path_factory):
    """analyze_scan results (no identification) for the synthetic scans."""
    from yq.box.analysis import analyze_scan
    res = {}
    for name, (folder, truth) in scans.items():
        res[name] = analyze_scan(folder, identify=False)
    return res
