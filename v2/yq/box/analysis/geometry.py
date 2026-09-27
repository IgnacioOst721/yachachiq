"""Camera model and the box geometry (CONTRACTS.md §4 coordinates).

Object frame: origin at the platter centre on its top surface, Z up, millimetres. The
object turns with the platter; at platter angle theta a point X fixed on the object is at
Rz(direction * theta) X in the box. A camera sees it at R_c Rz(direction*theta) X + t_c,
so the pose of any photo is a function of the platter angle (see `Camera.at_platter`).

Camera convention = OpenCV: x right, y down, z forward; K, distortion (k1,k2,p1,p2,k3).

Nominal values come from the CAD R1 (LEEME_FABRICACION.md / componentes.json) and are only
used until the calibration tools have measured the real box (calib.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# CAD frame: origin at the inner front-left floor corner, X right, Y back, Z up. The platter
# top is at Z=82 and its centre at X=Y=150, so object = CAD - CAD_ORIGIN.
CAD_ORIGIN = np.array([150.0, 150.0, 82.0])
PLATTER_RADIUS_MM = 90.0
MAX_OBJECT_RADIUS_MM = 75.0
MAX_OBJECT_HEIGHT_MM = 150.0
AIM_POINT = np.array([0.0, 0.0, 75.0])       # all cameras aim at mid-height of the max object

# Optical centres from the CAD R1 (item 5 of "Cambios resueltos"), in CAD coordinates.
CAD_CAMERAS = {
    "A": {"position": [150.0, 410.0, 251.632], "hfov_deg": 65.0, "size": (4656, 3496), "sensor": "IMX519 16 MP"},
    "B": {"position": [150.0, 305.0, 378.363], "hfov_deg": 65.0, "size": (4656, 3496), "sensor": "IMX519 16 MP"},
    "T": {"position": [-135.0, 150.0, 157.0], "hfov_deg": 57.0, "size": (160, 120), "sensor": "FLIR Lepton 3.5"},
}
# Raking LED centres (bounding-box centres of LED1..LED8 in componentes.json), CAD coordinates.
CAD_LEDS = {
    1: [74.55, 280.8, 150.4], 2: [270.65, 280.65, 150.4], 3: [21.3, 39.4, 150.35], 4: [21.2, 79.55, 150.4],
    5: [21.3, 260.6, 150.35], 6: [278.75, 54.45, 150.4], 7: [278.9, 150.0, 150.5], 8: [278.75, 245.55, 150.4],
}
CAD_UV = [250.0, 250.0, 223.4]        # UV source body centre, aimed 45 deg at the platter
CAD_HALOGEN = [246.0, 248.0, 292.0]   # GU10/MR16 envelope centre


def cad_to_object(p) -> np.ndarray:
    return np.asarray(p, dtype=float) - CAD_ORIGIN


def rotz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> tuple:
    """World->camera rotation R and translation t (OpenCV axes) for a camera at `eye` looking at `target`."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def rodrigues(v) -> np.ndarray:
    """Rotation vector -> matrix (no OpenCV needed)."""
    v = np.asarray(v, dtype=float).reshape(3)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def rotvec(R) -> np.ndarray:
    """Matrix -> rotation vector."""
    R = np.asarray(R, dtype=float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-6:
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        return V[:, np.argmax(w)] * th
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


@dataclass
class Camera:
    name: str
    K: np.ndarray
    R: np.ndarray                         # world(object at platter 0) -> camera
    t: np.ndarray
    width: int
    height: int
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    platter_deg: float = 0.0              # set by at_platter (informative)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self.t = np.asarray(self.t, dtype=float).reshape(3)
        d = np.zeros(5)
        dd = np.asarray(self.dist, dtype=float).ravel()[:5]
        d[: len(dd)] = dd
        self.dist = d

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def focal(self) -> float:
        return float((self.K[0, 0] + self.K[1, 1]) / 2)

    def at_platter(self, deg: float, direction: int = 1) -> "Camera":
        """Effective camera seeing the object frame when the platter is at `deg`."""
        return Camera(self.name, self.K, self.R @ rotz(direction * deg), self.t, self.width, self.height, self.dist, deg)

    def scaled(self, width: int, height: int) -> "Camera":
        """Same camera for an image resized to width x height (the Jetson may downscale photos)."""
        sx, sy = width / float(self.width), height / float(self.height)
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 1] *= sy
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        return Camera(self.name, K, self.R, self.t, int(width), int(height), self.dist, self.platter_deg)

    def to_cam(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float).reshape(-1, 3) @ self.R.T + self.t

    def project(self, X, return_depth: bool = False):
        """(N,3) object points -> (N,2) pixels (OpenCV distortion model)."""
        Xc = self.to_cam(X)
        z = Xc[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
        k1, k2, p1, p2, k3 = self.dist
        if np.any(self.dist):
            r2 = x * x + y * y
            rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
            xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        else:
            xd, yd = x, y
        uv = np.stack([self.K[0, 0] * xd + self.K[0, 1] * yd + self.K[0, 2], self.K[1, 1] * yd + self.K[1, 2]], axis=1)
        return (uv, z) if return_depth else uv

    def undistort_normalized(self, uv) -> np.ndarray:
        """Pixels -> undistorted normalized coordinates (x, y) with z=1."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if not np.any(self.dist):
            return np.stack([(uv[:, 0] - self.K[0, 2] - self.K[0, 1] * (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]) / self.K[0, 0],
                             (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]], axis=1)
        import cv2
        return cv2.undistortPoints(uv.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)

    def rays(self, uv) -> tuple:
        """Pixels -> (origin (3,), unit directions (N,3)) in the object frame."""
        xy = self.undistort_normalized(uv)
        d = np.concatenate([xy, np.ones((len(xy), 1))], axis=1) @ self.R   # R^T d_cam, row form
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return self.center, d

    def to_dict(self) -> dict:
        return {"name": self.name, "K": self.K.tolist(), "dist": self.dist.tolist(), "R": self.R.tolist(),
                "t": self.t.tolist(), "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(d.get("name", "?"), d["K"], d["R"], d["t"], int(d["width"]), int(d["height"]), d.get("dist", [0] * 5))


def nominal_camera(name: str, width: Optional[int] = None, height: Optional[int] = None) -> Camera:
    """CAD-nominal camera (pinhole, no distortion) aimed at AIM_POINT, landscape, image up = +Z."""
    spec = CAD_CAMERAS[name]
    w0, h0 = spec["size"]
    K = intrinsics_from_fov(w0, h0, spec["hfov_deg"])
    R, t = look_at(cad_to_object(spec["position"]), AIM_POINT)
    cam = Camera(name, K, R, t, w0, h0)
    if width and height and (width, height) != (w0, h0):
        cam = cam.scaled(width, height)
    return cam


def nominal_leds() -> dict:
    return {i: cad_to_object(p) for i, p in CAD_LEDS.items()}


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)
