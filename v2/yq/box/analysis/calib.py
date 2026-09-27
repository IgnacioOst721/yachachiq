"""Box calibration: file formats, loading with CAD fallbacks, and the calibration math entry points.

Files in config.CALIB_DIR (JSON, millimetres, OpenCV camera conventions, object frame of
CONTRACTS.md §4). meta.json "calibration" may name other files (absolute path or a file
name inside CALIB_DIR); otherwise these defaults are used:

  intrinsics.json   {"type": "yq-intrinsics-1", "cameras": {"A": {K, dist, width, height, rms_px, n_images, source, created}, "B": ..., "T": ...}}
  extrinsics.json   {"type": "yq-turntable-1", "direction": 1|-1, "cameras": {"A": {R, t, rms_px}, "B": ..., "T": ...},
                     "rms_px", "axis_tilt_deg", "n_views", "source", "created"}
  rti_lights.json   {"type": "yq-rti-lights-1", "camera": "B", "platter_deg": 0,
                     "lights": [{"index", "position_mm", "direction", "intensity", "source": "sphere"|"cad"}]}
  thermal_reg.json  {"type": "yq-thermal-reg-1", "K", "dist", "width", "height", "R", "t", "rms_px",
                     "homography": {"to_camera": "A", "platter_deg", "H", "rms_px"} | null, "source"}

`direction` is the sign of the platter rotation: +1 means platter_deg grows counter-clockwise
seen from above (right-hand rule about +Z). The turntable calibration measures it.
Anything missing falls back to the CAD R1 nominal geometry, with a Spanish warning.
The math lives in calib_charuco.py (intrinsics, turntable), calib_rti.py (chrome sphere),
calib_thermal.py (heated target) and calib_target.py (printable targets).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from yq.common import config

from .geometry import Camera, cad_to_object, nominal_camera, nominal_leds, unit, AIM_POINT, CAD_CAMERAS

FILES = {"intrinsics": "intrinsics.json", "extrinsics": "extrinsics.json", "rti_lights": "rti_lights.json",
         "thermal_reg": "thermal_reg.json"}


def calib_dir(path: Optional[Path] = None) -> Path:
    return Path(path) if path else Path(config.CALIB_DIR)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)
    return path


def _resolve(key: str, meta: Optional[dict], cdir: Path) -> Optional[Path]:
    name = ((meta or {}).get("calibration") or {}).get(key)
    cands = []
    if name:
        p = Path(str(name)).expanduser()
        cands += [p] if p.is_absolute() else [cdir / p, cdir / p.name]
    cands.append(cdir / FILES[key])
    for p in cands:
        if p.is_file():
            return p
    return None


def _read(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


@dataclass
class BoxCalibration:
    intrinsics: dict = field(default_factory=dict)      # name -> {K, dist, width, height, ...}
    extrinsics: dict = field(default_factory=dict)      # name -> {R, t, ...}
    direction: int = 1
    lights: list = field(default_factory=list)          # [{"index", "position_mm", "direction", "intensity", "source"}]
    rti_camera: str = "B"
    thermal: Optional[dict] = None
    sources: dict = field(default_factory=dict)         # what each part came from: file path or "cad"
    reproj_px: dict = field(default_factory=dict)       # name -> rms reprojection error (px) when calibrated
    warnings: list = field(default_factory=list)        # Spanish

    @classmethod
    def load(cls, meta: Optional[dict] = None, directory: Optional[Path] = None) -> "BoxCalibration":
        cdir = calib_dir(directory)
        cal = cls()
        intr = _read(_resolve("intrinsics", meta, cdir))
        extr = _read(_resolve("extrinsics", meta, cdir))
        rti = _read(_resolve("rti_lights", meta, cdir))
        th = _read(_resolve("thermal_reg", meta, cdir))
        cal.intrinsics = dict((intr or {}).get("cameras") or {})
        cal.extrinsics = dict((extr or {}).get("cameras") or {})
        cal.direction = int((extr or {}).get("direction", 1)) or 1
        for name in ("A", "B"):
            ok_i, ok_e = name in cal.intrinsics, name in cal.extrinsics
            cal.sources[name] = "calibrated" if ok_i and ok_e else ("partial" if ok_i or ok_e else "cad")
            if name in cal.intrinsics and "rms_px" in cal.intrinsics[name]:
                cal.reproj_px[name] = float(cal.intrinsics[name]["rms_px"])
            if name in cal.extrinsics and "rms_px" in cal.extrinsics[name]:
                cal.reproj_px[name] = max(cal.reproj_px.get(name, 0.0), float(cal.extrinsics[name]["rms_px"]))
        if any(cal.sources[n] != "calibrated" for n in ("A", "B")):
            cal.warnings.append("Cámaras sin calibrar por completo: se usa la geometría nominal del CAD; "
                                "las medidas pueden tener errores de varios milímetros.")
        if rti and rti.get("lights"):
            cal.lights = rti["lights"]
            cal.rti_camera = rti.get("camera", "B")
        else:
            cal.lights = [{"index": i, "position_mm": p.tolist(), "direction": unit(p - AIM_POINT * 0.4).tolist(),
                           "intensity": 1.0, "source": "cad"} for i, p in nominal_leds().items()]
            cal.warnings.append("Luces RTI sin calibrar (esfera cromada): se usan las posiciones nominales del CAD.")
        cal.thermal = th
        return cal

    def camera(self, name: str, width: Optional[int] = None, height: Optional[int] = None) -> Camera:
        """Camera `name` at platter 0, scaled to the photo size (width, height) when given."""
        if name == "T" and self.thermal and "K" in self.thermal:
            cam = Camera("T", self.thermal["K"], self.thermal.get("R", np.eye(3)), self.thermal.get("t", [0, 0, 0]),
                         int(self.thermal.get("width", 160)), int(self.thermal.get("height", 120)), self.thermal.get("dist", [0] * 5))
            if "R" not in self.thermal:
                nom = nominal_camera("T")
                cam = Camera("T", cam.K, nom.R, nom.t, cam.width, cam.height, cam.dist)
        else:
            nom = nominal_camera(name)
            i = self.intrinsics.get(name)
            e = self.extrinsics.get(name)
            K, dist, w, h = (i["K"], i.get("dist", [0] * 5), int(i["width"]), int(i["height"])) if i else \
                (nom.K, nom.dist, nom.width, nom.height)
            R, t = (e["R"], e["t"]) if e else (nom.R, nom.t)
            cam = Camera(name, K, R, t, w, h, dist)
        if width and height and (int(width), int(height)) != (cam.width, cam.height):
            if abs(width / float(height) - cam.width / float(cam.height)) > 0.02:
                self.warnings.append("La foto de la cámara %s (%dx%d) no tiene la proporción de la calibración (%dx%d)."
                                     % (name, width, height, cam.width, cam.height))
            cam = cam.scaled(int(width), int(height))
        return cam

    def view(self, name: str, platter_deg: float, width: Optional[int] = None, height: Optional[int] = None) -> Camera:
        return self.camera(name, width, height).at_platter(platter_deg, self.direction)

    def light_positions(self) -> np.ndarray:
        return np.array([l["position_mm"] for l in sorted(self.lights, key=lambda l: l["index"])], dtype=float)

    def calibrated(self) -> bool:
        return all(self.sources.get(n) == "calibrated" for n in ("A", "B"))


def save_intrinsics(results: dict, directory: Optional[Path] = None, merge: bool = True) -> Path:
    """results: name -> {K, dist, width, height, rms_px, n_images, source}."""
    path = calib_dir(directory) / FILES["intrinsics"]
    data = _read(path) if merge else None
    data = data or {"type": "yq-intrinsics-1", "cameras": {}}
    for name, r in results.items():
        data["cameras"][name] = dict(r, created=now())
    return write_json(path, data)


def save_extrinsics(cameras: dict, direction: int, extra: Optional[dict] = None, directory: Optional[Path] = None,
                    merge: bool = True) -> Path:
    path = calib_dir(directory) / FILES["extrinsics"]
    data = _read(path) if merge else None
    data = data or {"type": "yq-turntable-1", "cameras": {}}
    data["direction"] = int(direction)
    for name, c in cameras.items():
        data["cameras"][name] = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
    data.update(extra or {})
    data["created"] = now()
    return write_json(path, data)


def save_rti_lights(lights: list, camera: str = "B", platter_deg: float = 0.0, directory: Optional[Path] = None,
                    extra: Optional[dict] = None) -> Path:
    data = {"type": "yq-rti-lights-1", "camera": camera, "platter_deg": platter_deg, "lights": lights, "created": now()}
    data.update(extra or {})
    return write_json(calib_dir(directory) / FILES["rti_lights"], data)


def save_thermal(reg: dict, directory: Optional[Path] = None) -> Path:
    data = {"type": "yq-thermal-reg-1", "created": now()}
    data.update({k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in reg.items()})
    return write_json(calib_dir(directory) / FILES["thermal_reg"], data)


def camera_record(cam: Camera, rms_px: float = 0.0, n_images: int = 0, source: str = "charuco") -> tuple:
    """(intrinsics entry, extrinsics entry) for a Camera."""
    return ({"K": cam.K.tolist(), "dist": cam.dist.tolist(), "width": cam.width, "height": cam.height,
             "rms_px": float(rms_px), "n_images": int(n_images), "source": source},
            {"R": cam.R.tolist(), "t": cam.t.tolist(), "rms_px": float(rms_px)})


def nominal_summary() -> dict:
    """Human-readable nominal geometry (docs, UI)."""
    return {n: {"position_mm": cad_to_object(s["position"]).round(1).tolist(), "hfov_deg": s["hfov_deg"], "sensor": s["sensor"]}
            for n, s in CAD_CAMERAS.items()}
