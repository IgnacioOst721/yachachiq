"""Read a scan folder written by BOX-CAPTURE (CONTRACTS.md §4). Tolerant: missing parts become warnings."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_PHOTO = re.compile(r"^cam([A-Za-z0-9]+)_(\d{3})(?:_(\d+))?\.(jpg|jpeg|png)$", re.I)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load_image(path: Path) -> Optional[np.ndarray]:
    """RGB uint8 (or None)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else img[:, :, ::-1].copy()


@dataclass
class Photo:
    camera: str
    platter_deg: float
    path: Path


@dataclass
class Scan:
    folder: Path
    meta: dict = field(default_factory=dict)
    weight: Optional[dict] = None
    photos: list = field(default_factory=list)             # [Photo]
    backgrounds: dict = field(default_factory=dict)        # camera -> Path
    rti: Optional[dict] = None                             # {"images": {index: Path}, "ambient": Path, "lights": dict}
    uv: Optional[dict] = None                              # {"uv": Path, "visible": Path, "dark": Path, "exposure": dict}
    thermal: Optional[dict] = None                         # {"sequence": Path, "times": Path, "meta": dict}
    warnings: list = field(default_factory=list)

    @property
    def scan_id(self) -> str:
        return self.meta.get("scan_id") or self.folder.name

    def cameras(self) -> list:
        return sorted({p.camera for p in self.photos})


def _validate_with_capture(folder: Path, warnings: list) -> None:
    """Use BOX-CAPTURE's validator when it exists (yq/box/layout.py); never fail because of it."""
    try:
        from yq.box.layout import validate_scan_folder  # owned by BOX-CAPTURE
    except Exception:
        return
    try:
        res = validate_scan_folder(folder)
    except Exception as e:
        warnings.append("Validación de la carpeta: %s" % e)
        return
    problems = []
    if isinstance(res, dict):
        problems = res.get("problems_es") or res.get("problems") or res.get("errors") or []
    elif isinstance(res, (list, tuple)):
        problems = list(res)
    for p in problems:
        warnings.append("Validación de la carpeta: %s" % p)


def load_scan(folder: Path, validate: bool = True) -> Scan:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError("no existe la carpeta del escaneo: %s" % folder)
    s = Scan(folder=folder, meta=read_json(folder / "meta.json", {}) or {})
    if not s.meta:
        s.warnings.append("Falta meta.json: se usan valores por defecto.")
    if validate:
        _validate_with_capture(folder, s.warnings)
    s.weight = read_json(folder / "weight.json")
    pg = folder / "photogrammetry"
    if pg.is_dir():
        poses = read_json(pg / "poses.json", {}) or {}
        for p in sorted(pg.iterdir()):
            m = _PHOTO.match(p.name)
            if not m:
                continue
            deg = poses.get(p.name, {}).get("platter_deg")
            if deg is None:
                deg = float(m.group(2))
                if poses:
                    s.warnings.append("%s no está en poses.json: se usa el ángulo del nombre (%g°)." % (p.name, deg))
            s.photos.append(Photo(m.group(1).upper(), float(deg), p))
        for p in pg.glob("background_cam*.*"):
            s.backgrounds[p.stem.replace("background_cam", "").upper()] = p
    rti = folder / "rti"
    if rti.is_dir():
        imgs = {}
        for p in rti.iterdir():
            m = re.match(r"^led(\d+)\.(jpg|jpeg|png)$", p.name, re.I)
            if m:
                imgs[int(m.group(1))] = p
        amb = next((p for p in rti.glob("ambient.*")), None)
        if imgs:
            lights = read_json(rti / "lights.json", {}) or {}
            if isinstance(lights, list):      # CONTRACTS §4 / BOX-CAPTURE: a plain list of the 8 lights
                lights = {"lights": lights}
            s.rti = {"images": imgs, "ambient": amb, "lights": lights}
    uv = folder / "uv"
    if uv.is_dir():
        d = {k: next(iter(sorted(uv.glob(k + ".*"))), None) for k in ("uv", "visible", "dark")}
        if d["uv"] is not None and d["visible"] is not None:
            d["exposure"] = read_json(uv / "exposure.json", {}) or {}
            s.uv = d
    th = folder / "thermal"
    if (th / "sequence.npy").exists() and (th / "times.npy").exists():
        s.thermal = {"sequence": th / "sequence.npy", "times": th / "times.npy", "meta": read_json(th / "meta.json", {}) or {}}
    return s
