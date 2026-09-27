"""Write complete synthetic scan folders (CONTRACTS.md §4) and calibration photos from ground truth.

    truth = make_scan(folder, Cylinder(40, 100), size=(776, 582))
    result = analyze_scan(folder)          # compare with truth

Everything is rendered from the "true" cameras (CAD nominal unless given), so an analysis that
uses the matching calibration must recover the known sizes, volume, normals and defects.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from . import calib as calmod
from .calib_charuco import TURNTABLE_BOARD, BoardSpec, R_FLIP
from .geometry import Camera, nominal_camera, nominal_leds, rodrigues
from .synthetic import Shape, render_view, shade, to_uint8, trace
from .synthetic_physics import simulate_thermal, uv_images


def _save(path: Path, img: np.ndarray, quality: int = 95) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3:
        img = img[:, :, ::-1]
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix == ".jpg" else [])


def render_board_pose(view: Camera, spec: BoardSpec, Rb, b, ppm: float = 12.0, ss: int = 2, rng=None,
                      background=None) -> np.ndarray:
    """Gray photo of the board whose frame maps to the camera's world by X = Rb @ Xboard + b."""
    import cv2
    board = spec.board()
    W, H = spec.size_mm
    tex = board.generateImage((int(round(W * ppm)), int(round(H * ppm))), marginSize=0, borderBits=1)
    Rb, b = np.asarray(Rb, float), np.asarray(b, float)
    off = (np.arange(ss) + 0.5) / ss - 0.5
    u = (np.arange(view.width)[:, None] + off).ravel()
    v = (np.arange(view.height)[:, None] + off).ravel()
    uu, vv = np.meshgrid(u, v)
    o, d = view.rays(np.stack([uu.ravel(), vv.ravel()], axis=1))
    n = Rb[:, 2]
    t = ((b - o) @ n) / np.where(np.abs(d @ n) < 1e-12, 1e-12, d @ n)
    P = o[None] + t[:, None] * d
    Xb = (P - b) @ Rb                                  # board coordinates (mm)
    mx = (Xb[:, 0] * ppm - 0.5).astype(np.float32).reshape(vv.shape)
    my = (Xb[:, 1] * ppm - 0.5).astype(np.float32).reshape(vv.shape)
    img = cv2.remap(tex.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    inside = (img >= 0) & (t.reshape(vv.shape) > 0)
    if background is None:
        pl = np.hypot(P[:, 0], P[:, 1]).reshape(vv.shape) <= 90.0
        background = np.where(pl, 25.0, 8.0)
    img = np.where(inside, img * 0.85 + 20, background)
    img = img.reshape(view.height, ss, view.width, ss).mean(axis=(1, 3))
    if rng is not None:
        img = img + rng.normal(0, 1.5, img.shape)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def render_board_view(cam: Camera, spec: BoardSpec = TURNTABLE_BOARD, platter_deg: float = 0.0, direction: int = 1,
                      board_xy=(-54.0, 54.0), tilt=(0.0, 0.0), thickness: float = 0.3, ppm: float = 12.0, ss: int = 2,
                      rng=None) -> np.ndarray:
    """Gray photo of the flat ChArUco board lying on the platter (x right, y towards -Y world)."""
    Rb = rodrigues([tilt[0], tilt[1], 0.0]) @ R_FLIP
    b = np.array([board_xy[0], board_xy[1], thickness])
    return render_board_pose(cam.at_platter(platter_deg, direction), spec, Rb, b, ppm, ss, rng)


def handheld_board_poses(n: int, seed: int = 0, distance=(230.0, 330.0), tilt_deg: float = 35.0) -> list:
    """Random board poses in front of a camera (camera frame): [(Rb, b)] for intrinsics photos."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        R = rodrigues(np.radians(rng.uniform(-tilt_deg, tilt_deg, 3)) * [1, 1, 0.3])
        z = rng.uniform(*distance)
        c = np.array([rng.uniform(-0.25, 0.25) * z, rng.uniform(-0.18, 0.18) * z, z])
        out.append((R, c))
    return out


def true_cameras(size=(776, 582), perturb: float = 0.0, distortion: bool = True, seed: int = 0) -> dict:
    """The 'real' cameras of a synthetic box: CAD nominal, optionally perturbed (mm / deg) with lens distortion."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("A", "B"):
        c = nominal_camera(name, *size)
        R, t = c.R, c.t
        if perturb:
            R = rodrigues(rng.normal(0, math.radians(perturb), 3)) @ R
            t = t + rng.normal(0, perturb, 3)
        K = c.K.copy()
        if perturb:
            K[0, 0] *= 1 + rng.normal(0, 0.01)
            K[1, 1] = K[0, 0] * (1 + rng.normal(0, 0.001))
            K[0, 2] += rng.normal(0, 4)
            K[1, 2] += rng.normal(0, 4)
        dist = [-0.08, 0.05, 0.0, 0.0, 0.0] if distortion else [0] * 5
        out[name] = Camera(name, K, R, t, size[0], size[1], dist)
    return out


def write_calibration(cams: dict, directory: Optional[Path] = None, lights: Optional[dict] = None, direction: int = 1) -> None:
    """Store the true cameras as if the calibration tools had measured them (tests of the analyses)."""
    intr, extr = {}, {}
    for n, c in cams.items():
        i, e = calmod.camera_record(c, rms_px=0.3, n_images=20, source="synthetic-truth")
        intr[n], extr[n] = i, e
    calmod.save_intrinsics(intr, directory, merge=False)
    calmod.save_extrinsics(extr, direction, {"rms_px": 0.3, "source": "synthetic-truth"}, directory, merge=False)
    if lights is not None:
        calmod.save_rti_lights([{"index": i, "position_mm": list(map(float, p)), "direction": list(map(float, p / np.linalg.norm(p))),
                                 "intensity": 1.0, "source": "synthetic-truth"} for i, p in lights.items()], "B", 0.0, directory)


def make_scan(folder: Path, shape: Shape, size=(776, 582), angles=None, cams: Optional[dict] = None, mass_g: float = 850.0,
              rti: bool = True, uv: bool = True, thermal: bool = True, calib_dir: Optional[Path] = None,
              write_calib: bool = True, direction: int = 1, uv_zone=None, defect=None, seed: int = 0) -> dict:
    """Render a full scan of `shape`. Returns the ground truth (dims, volume, zones, defects...)."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    angles = list(range(0, 360, 15)) if angles is None else list(angles)
    cams = cams or true_cameras(size)
    leds = nominal_leds()
    if write_calib:
        write_calibration(cams, calib_dir, leds, direction)
    scan_id = folder.name
    photo = folder / "photogrammetry"
    poses = {}
    for name, cam in cams.items():
        for a in angles:
            fn = "cam%s_%03d.jpg" % (name, a)
            _save(photo / fn, render_view(shape, cam.at_platter(a, direction), noise=0.004, rng=rng))
            poses[fn] = {"platter_deg": float(a)}
        _save(photo / ("background_cam%s.jpg" % name), render_view(None, cam, noise=0.004, rng=rng))
    (photo / "poses.json").write_text(json.dumps(poses, indent=1))
    truth = {"shape": shape.name, "dims": {k: float(v) for k, v in shape.dims().items()}, "volume_mm3": float(shape.volume()),
             "mass_g": mass_g}
    weight = {"grams": mass_g, "sigma_g": 0.5, "samples": [mass_g + float(x) for x in rng.normal(0, 0.5, 10)], "tare_g": 1150.0,
              "stable": True, "calibration_factor": 412.7}
    (folder / "weight.json").write_text(json.dumps(weight, indent=1))
    analyses = ["weight", "photogrammetry"]
    if rti:
        analyses.append("rti")
        camB = cams["B"]
        tr = trace(shape, camB, ss=2)
        lights = []
        for i, p in sorted(leds.items()):
            img = shade(tr, shape, [{"pos": p, "power": 0.9, "axis": np.array([0, 0, 40.0]) - p}], ambient=0.01, specular=0.25)
            _save(folder / "rti" / ("led%d.jpg" % i), to_uint8(img, noise=0.003, rng=rng))
            lights.append({"index": i, "position_mm": p.tolist(), "direction": (p / np.linalg.norm(p)).tolist()})
        _save(folder / "rti" / "ambient.jpg", to_uint8(shade(tr, shape, [], ambient=0.01), noise=0.003, rng=rng))
        (folder / "rti" / "lights.json").write_text(json.dumps({"camera": "B", "platter_deg": 0.0, "lights": lights}, indent=1))
    if uv:
        analyses.append("uv")
        imgs, zone = uv_images(shape, cams["A"], zone=uv_zone, rng=rng)
        for k, im in imgs.items():
            _save(folder / "uv" / ("%s.jpg" % k), im)
        (folder / "uv" / "exposure.json").write_text(json.dumps({"camera": "A", "platter_deg": 0.0, "uv_exposure_us": 200000,
                                                                "visible_exposure_us": 20000, "dark_exposure_us": 200000}))
        truth["uv_zone"] = zone
    if thermal:
        analyses.append("thermal")
        seq, times, tmeta, dtruth = simulate_thermal(shape, nominal_camera("T"), defect=defect, rng=rng)
        (folder / "thermal").mkdir(parents=True, exist_ok=True)
        np.save(folder / "thermal" / "sequence.npy", seq)
        np.save(folder / "thermal" / "times.npy", times)
        (folder / "thermal" / "meta.json").write_text(json.dumps(tmeta, indent=1))
        truth["defect"] = dtruth
    meta = {"scan_id": scan_id, "profile": "standard", "analyses": analyses, "started": 0.0, "finished": 0.0,
            "box": {"firmware": "synthetic", "platter_deg": 0.0},
            "cameras": {n: {"id": "synthetic-%s" % n, "resolution": [c.width, c.height], "focus": 0, "exposure_us": 20000,
                            "gain": 1.0, "wb_k": 5000} for n, c in cams.items()},
            "calibration": {k: (str(Path(calib_dir) / f) if calib_dir else f) for k, f in calmod.FILES.items()},
            "door_closed": True, "warnings": [], "synthetic": True}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    (folder / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth
