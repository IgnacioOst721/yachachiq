"""Camera model, ChArUco intrinsics, turntable extrinsics, calibration files and CAD fallbacks."""
from __future__ import annotations

import math

import numpy as np
import pytest

from yq.box.analysis.geometry import Camera, nominal_camera, rodrigues, rotvec


def test_projection_roundtrip_and_scaling():
    cam = nominal_camera("A", 776, 582)
    X = np.array([[10.0, -20.0, 50.0], [0, 0, 75.0], [-40, 30, 5]])
    uv = cam.project(X)
    o, d = cam.rays(uv)
    for x, di in zip(X, d):                          # the ray through the pixel passes through the point
        assert np.linalg.norm(np.cross(x - o, di)) < 1e-6
    assert np.allclose(cam.project([[0, 0, 75.0]])[0], [387.5, 290.5], atol=0.5)   # aimed at the object centre
    big = cam.scaled(1552, 1164)
    assert np.allclose(big.project(X), (uv + 0.5) * 2 - 0.5, atol=1e-6)
    v = cam.at_platter(90.0)                         # a point at +x seen at platter 90 == point at +y at platter 0
    assert np.allclose(v.project([[30.0, 0, 20]]), cam.project([[0, 30.0, 20]]), atol=1e-6)


def test_intrinsics_from_synthetic_charuco():
    from yq.box.analysis.calib_charuco import INTRINSICS_BOARD, calibrate_intrinsics
    from yq.box.analysis.synthetic_scan import handheld_board_poses, render_board_pose
    K = np.array([[900.0, 0, 395.0], [0, 902.0, 288.0], [0, 0, 1]])
    cam = Camera("X", K, np.eye(3), np.zeros(3), 776, 582, [-0.12, 0.08, 0, 0, 0])
    W, H = INTRINSICS_BOARD.size_mm
    imgs = []
    for R, c in handheld_board_poses(14, seed=2, distance=(260, 360)):
        b = c - R @ np.array([W / 2, H / 2, 0.0])     # board centre at c
        imgs.append(render_board_pose(cam, INTRINSICS_BOARD, R, b, rng=np.random.default_rng(1), background=10.0))
    r = calibrate_intrinsics(imgs)
    assert r["n_images"] >= 10 and r["rms_px"] < 0.5
    Kf = np.array(r["K"])
    assert abs(Kf[0, 0] - 900) / 900 < 0.01 and abs(Kf[0, 2] - 395) < 4 and abs(Kf[1, 2] - 288) < 4
    assert r["dist"][0] == pytest.approx(-0.12, abs=0.03)


def test_turntable_recovers_poses_and_direction():
    from yq.box.analysis.calib_charuco import calibrate_turntable
    from yq.box.analysis.synthetic_scan import render_board_view, true_cameras
    cams = true_cameras((776, 582), perturb=3.0, seed=5)
    rng = np.random.default_rng(0)
    views = [(n, a, render_board_view(cams[n], platter_deg=a, direction=-1, board_xy=(-50, 57), rng=rng))
             for n in "AB" for a in range(0, 360, 30)]
    r = calibrate_turntable(views, cams)
    assert r["direction"] == -1 and r["rms_px"] < 0.5
    for n in "AB":
        R, t = np.array(r["cameras"][n]["R"]), np.array(r["cameras"][n]["t"])
        assert np.linalg.norm(-R.T @ t - cams[n].center) < 0.5                     # mm
        assert math.degrees(np.linalg.norm(rotvec(R @ cams[n].R.T))) < 0.1       # deg


def test_calibration_files_and_fallback(tmp_path):
    from yq.box.analysis import calib
    c0 = calib.BoxCalibration.load(directory=tmp_path)
    assert c0.sources == {"A": "cad", "B": "cad"} and c0.warnings and len(c0.lights) == 8
    cam = nominal_camera("B")
    i, e = calib.camera_record(cam, 0.3, 20)
    calib.save_intrinsics({"B": i}, tmp_path)
    calib.save_extrinsics({"B": e}, -1, {"rms_px": 0.3}, tmp_path)
    c1 = calib.BoxCalibration.load(directory=tmp_path)
    assert c1.sources["B"] == "calibrated" and c1.direction == -1
    v = c1.view("B", 30.0, 1164, 873)
    assert v.width == 1164 and np.allclose(v.R, cam.R @ np.array(rodrigues([0, 0, math.radians(-30)])))


def test_targets(tmp_path):
    from PIL import Image
    from yq.box.analysis.calib_target import make_targets
    made = make_targets(tmp_path)
    assert set(made) == {"charuco_intrinsics", "charuco_plato", "termico_ajedrez"}
    im = Image.open(made["charuco_plato"][0])
    assert im.size == (round(210 / 25.4 * 600), round(297 / 25.4 * 600))
    assert 'width="210mm"' in open(made["charuco_plato"][2]).read()
