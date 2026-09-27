"""Scan folder layout (CONTRACTS.md §4): names, and a validator both BOX-CAPTURE
and BOX-ANALYSIS use.

    from yq.box.layout import validate_scan_folder
    problems = validate_scan_folder("~/yq-data/scans/scan-...")   # [] = OK

Light by design: JSON is parsed, images are checked for the JPEG signature,
.npy headers are read without loading the arrays.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from yq.common.contracts import ANALYSES, SCAN_PROFILES

META_KEYS = ("scan_id", "profile", "analyses", "started", "finished", "box", "cameras", "calibration",
             "door_closed", "warnings")
CALIB_KEYS = ("intrinsics", "extrinsics", "rti_lights", "thermal_reg")
WEIGHT_KEYS = ("grams", "sigma_g", "samples", "tare_g", "stable", "calibration_factor")
THERMAL_META_KEYS = ("heat_on_s", "heat_off_s", "halogen", "ambient_c", "fps")
RESULT_KEYS = ("scan_id", "folder", "profile", "started", "finished", "measurements", "artifacts", "findings",
               "warnings", "ok")
PHOTO_RE = re.compile(r"^cam([AB])_(\d{3})\.jpg$")
RTI_FILES = ["led%d.jpg" % i for i in range(1, 9)] + ["ambient.jpg"]
UV_FILES = ["uv.jpg", "visible.jpg", "dark.jpg"]
THERMAL_SHAPE = (120, 160)


def photo_name(cam: str, deg: float) -> str:
    """camA_010.jpg: nominal platter angle in whole degrees, 3 digits."""
    return "cam%s_%03d.jpg" % (cam, int(round(deg)) % 360)


def _json(path: Path, problems: list):
    if not path.is_file():
        problems.append("falta %s" % path.name)
        return None
    try:
        return json.loads(path.read_text())
    except ValueError as e:
        problems.append("%s no es JSON válido: %s" % (path.name, e))
        return None


def _jpeg(path: Path, problems: list, rel: str) -> None:
    if not path.is_file():
        problems.append("falta %s" % rel)
        return
    with open(path, "rb") as fh:
        if fh.read(3) != b"\xff\xd8\xff":
            problems.append("%s no es un JPEG" % rel)


def _npy_header(path: Path):
    import numpy as np
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(fh)
        return np.lib.format.read_array_header_2_0(fh)


def expected_stops(profile: str) -> int:
    from . import settings as S
    return S.PROFILES[profile]["stops"]


def validate_scan_folder(path, analyses: Optional[list] = None, require_result: bool = False) -> list:
    """Problems (Spanish, human readable) with the scan folder; [] when it follows §4.
    `analyses` defaults to meta.json's list; "identify" needs no captured data."""
    root = Path(path).expanduser()
    problems: list = []
    if not root.is_dir():
        return ["la carpeta %s no existe" % root]
    meta = _json(root / "meta.json", problems)
    if isinstance(meta, dict):
        for k in META_KEYS:
            if k not in meta:
                problems.append("meta.json: falta la clave %s" % k)
        if meta.get("scan_id") not in (None, root.name):
            problems.append("meta.json: scan_id %r no coincide con la carpeta %r" % (meta.get("scan_id"), root.name))
        if meta.get("profile") not in SCAN_PROFILES:
            problems.append("meta.json: perfil desconocido %r" % meta.get("profile"))
        bad = [a for a in meta.get("analyses") or [] if a not in ANALYSES]
        if bad:
            problems.append("meta.json: análisis desconocidos %s" % bad)
        box = meta.get("box") or {}
        for k in ("firmware", "platter_deg"):
            if k not in box:
                problems.append("meta.json: box.%s falta" % k)
        for k in CALIB_KEYS:
            if k not in (meta.get("calibration") or {}):
                problems.append("meta.json: calibration.%s falta" % k)
        if not isinstance(meta.get("warnings", []), list):
            problems.append("meta.json: warnings debe ser una lista")
    wanted = list(analyses if analyses is not None else (meta or {}).get("analyses") or ANALYSES)
    profile = (meta or {}).get("profile", "standard")

    if "weight" in wanted:
        w = _json(root / "weight.json", problems)
        if isinstance(w, dict):
            for k in WEIGHT_KEYS:
                if k not in w:
                    problems.append("weight.json: falta %s" % k)
            if not isinstance(w.get("grams"), (int, float)):
                problems.append("weight.json: grams debe ser un número")
            if not isinstance(w.get("samples", []), list):
                problems.append("weight.json: samples debe ser una lista")

    if "photogrammetry" in wanted:
        _check_photogrammetry(root / "photogrammetry", profile, problems)
    if "rti" in wanted:
        d = root / "rti"
        for f in RTI_FILES:
            _jpeg(d / f, problems, "rti/" + f)
        lights = _json(d / "lights.json", problems)
        if lights is not None:
            _check_lights(lights, problems)
    if "uv" in wanted:
        d = root / "uv"
        for f in UV_FILES:
            _jpeg(d / f, problems, "uv/" + f)
        _json(d / "exposure.json", problems)
    if "thermal" in wanted:
        _check_thermal(root / "thermal", problems)
    if require_result:
        res = _json(root / "result.json", problems)
        if isinstance(res, dict):
            for k in RESULT_KEYS:
                if k not in res:
                    problems.append("result.json: falta %s" % k)
    return problems


def _check_photogrammetry(d: Path, profile: str, problems: list) -> None:
    if not d.is_dir():
        problems.append("falta la carpeta photogrammetry/")
        return
    poses = _json(d / "poses.json", problems) or {}
    images = sorted(p.name for p in d.glob("cam*.jpg") if PHOTO_RE.match(p.name))
    per_cam = {"A": 0, "B": 0}
    for name in images:
        per_cam[PHOTO_RE.match(name).group(1)] += 1
        _jpeg(d / name, problems, "photogrammetry/" + name)
        pose = poses.get(name)
        if not isinstance(pose, dict) or not isinstance(pose.get("platter_deg"), (int, float)):
            problems.append("poses.json: falta platter_deg para %s" % name)
    for name in poses:
        if not (d / name).is_file():
            problems.append("poses.json menciona %s pero no existe" % name)
    try:
        want = expected_stops(profile)
    except KeyError:
        want = None
    for cam, n in per_cam.items():
        if n == 0:
            problems.append("photogrammetry/: no hay fotos de la cámara %s" % cam)
        elif want and n != want:
            problems.append("photogrammetry/: cámara %s tiene %d fotos, el perfil %s pide %d" % (cam, n, profile, want))
    for cam in ("A", "B"):
        bg = d / ("background_cam%s.jpg" % cam)
        if bg.exists():
            _jpeg(bg, problems, "photogrammetry/" + bg.name)


def _check_lights(lights, problems: list) -> None:
    if not isinstance(lights, list) or len(lights) != 8:
        problems.append("rti/lights.json debe tener 8 luces")
        return
    for i, L in enumerate(lights, start=1):
        if L.get("index") != i:
            problems.append("rti/lights.json: índice %s en la posición %d" % (L.get("index"), i))
        pos, dirn = L.get("position_mm"), L.get("direction")
        if not (isinstance(pos, list) and len(pos) == 3):
            problems.append("rti/lights.json: position_mm de la luz %d" % i)
        if not (isinstance(dirn, list) and len(dirn) == 3) or abs(sum(c * c for c in dirn) - 1.0) > 1e-3:
            problems.append("rti/lights.json: direction de la luz %d no es un vector unitario" % i)


def _check_thermal(d: Path, problems: list) -> None:
    seq, times = d / "sequence.npy", d / "times.npy"
    shape_t = None
    for f in (seq, times):
        if not f.is_file():
            problems.append("falta thermal/%s" % f.name)
    if seq.is_file():
        shape, _fortran, dtype = _npy_header(seq)
        if len(shape) != 3 or tuple(shape[1:]) != THERMAL_SHAPE or str(dtype) != "float32":
            problems.append("thermal/sequence.npy es %s %s, se espera T x 120 x 160 float32" % (shape, dtype))
        shape_t = shape[0] if shape else None
    if times.is_file():
        tshape, _f, _dt = _npy_header(times)
        if shape_t is not None and tuple(tshape) != (shape_t,):
            problems.append("thermal/times.npy tiene forma %s, se espera (%s,)" % (tshape, shape_t))
    meta = _json(d / "meta.json", problems)
    if isinstance(meta, dict):
        for k in THERMAL_META_KEYS:
            if k not in meta:
                problems.append("thermal/meta.json: falta %s" % k)
