"""Synthetic box camera: renders a textured vessel on the black platter, rotated
by the simulated platter angle and lit by whichever simulated light is on.

Geometry is the same as the real box (settings.CAMERA_GEOMETRY, light
positions from the CAD), so images from camera A/B, RTI and UV look like what
the real cameras should see. Texture is deterministic, so features stay put
on the object as it turns (usable for photogrammetry tests).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import settings as S

PROFILE_Z = np.array([0.0, 10, 40, 70, 95, 105, 120])
PROFILE_R = np.array([28.0, 42, 50, 44, 22, 18, 21])
BEST_FOCUS = 300
MARKERS_DEG = (3, 17, 41, 58, 90, 107, 131, 160, 178, 205, 227, 250, 262, 290, 311, 338)


def _norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _dot(a, v):
    """Row-wise a . v without BLAS (Accelerate raises spurious FP warnings on matmul)."""
    return a[:, 0] * v[0] + a[:, 1] * v[1] + a[:, 2] * v[2]


def _obj(p):
    return np.array([a - b for a, b in zip(p, S.CAD_TO_OBJECT)], dtype=np.float64)


class VesselModel:
    def __init__(self, n_theta: int = 1400, n_z: int = 420):
        th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        z = np.linspace(0.5, PROFILE_Z[-1], n_z)
        self.theta, self.z = np.meshgrid(th, z)                      # (n_z, n_theta)
        self.r = np.interp(self.z, PROFILE_Z, PROFILE_R)
        drdz = np.gradient(np.interp(z, PROFILE_Z, PROFILE_R), z)[:, None] * np.ones_like(self.theta)
        self.drdz = drdz
        self.albedo, self.uv = self._texture()

    def _texture(self):
        th, z = self.theta, self.z
        alb = np.empty(th.shape + (3,))
        alb[:] = (0.60, 0.38, 0.25)                                   # terracotta
        band = ((z > 20) & (z < 26)) | ((z > 85) & (z < 90))
        alb[band] = (0.20, 0.10, 0.06)
        warp = th + 0.35 * np.sin(3 * th)                             # irregular (not rotationally periodic)
        fret = ((z > 40) & (z < 75)) & ((((warp * 7 / (2 * np.pi)) % 1) < 0.5) ^ ((((z - 40) / 7) % 2) < 1))
        fret &= ~((th > 0.2) & (th < 1.3))                            # panel for the figure
        alb[fret] = (0.85, 0.78, 0.62)
        fig = ((th - 0.75) / 0.38) ** 2 + ((z - 58) / 11) ** 2 < 1     # a painted figure on one side only
        alb[fig] = (0.55, 0.15, 0.08)
        alb[((th - 0.95) / 0.06) ** 2 + ((z - 62) / 2.2) ** 2 < 1] = (0.9, 0.85, 0.7)
        ti = (th * 1000).astype(np.int64)
        zi = (z * 10).astype(np.int64)
        h = (ti * 73856093 ^ zi * 19349663) % 1000 / 1000.0          # deterministic grain
        alb *= (0.88 + 0.24 * h)[..., None]
        uv = np.full(th.shape, 0.06)
        uv[(th > 2.2) & (th < 2.9) & (z > 45) & (z < 80)] = 1.0       # modern restoration fluoresces
        return np.clip(alb, 0, 1), uv

    def points(self, platter_deg: float):
        phi = self.theta + math.radians(platter_deg)
        c, s = np.cos(phi), np.sin(phi)
        pts = np.stack([self.r * c, self.r * s, self.z], axis=-1).reshape(-1, 3)
        nrm = _norm(np.stack([c, s, -self.drdz], axis=-1)).reshape(-1, 3)
        # dark interior seen through the mouth: a disc just below the rim
        rho = np.sqrt(np.linspace(0, 1, 60))[:, None] * (PROFILE_R[-1] - 0.5)
        cap = np.stack([rho * np.cos(phi[:1]), rho * np.sin(phi[:1]), np.full_like(rho * phi[:1], PROFILE_Z[-1] - 3)],
                       axis=-1).reshape(-1, 3)
        cap_n = np.tile([0.0, 0.0, 1.0], (len(cap), 1))
        return (np.vstack([pts, cap]), np.vstack([nrm, cap_n]),
                np.vstack([self.albedo.reshape(-1, 3), np.full((len(cap), 3), 0.004)]),
                np.concatenate([self.uv.reshape(-1), np.zeros(len(cap))]))


_VESSELS: dict = {}


def _vessel(width: int) -> VesselModel:
    """Surface samples dense enough for the image width (fewer for small test images)."""
    s = round(min(1.0, max(0.3, width / 1600.0 * 1.3)), 2)
    if s not in _VESSELS:
        _VESSELS[s] = VesselModel(int(1400 * s), int(420 * s))
    return _VESSELS[s]


def camera_basis(pos, aim):
    f = _norm(np.asarray(aim, float) - np.asarray(pos, float))
    r = _norm(np.cross(f, [0.0, 0.0, 1.0]))
    d = np.cross(f, r)
    return f, r, d


def _light_terms(pts, nrm, lights: dict):
    """Linear radiance factors (N,3) from the visible lights that are on."""
    rad = np.zeros((len(pts), 3))
    cob = lights.get("cob", 0.0)
    if cob > 0:
        top = np.clip(nrm[:, 2], 0, 1)
        rad += cob * (0.45 + 0.55 * top)[:, None] * np.array([1.0, 0.98, 0.95])
    for i, p in enumerate(S.RAKE_POSITIONS_CAD, start=1):
        lv = lights.get("rake%d" % i, 0.0)
        if lv > 0:
            L = _obj(p) - pts
            dist = np.linalg.norm(L, axis=1)
            ndl = np.clip(np.sum(nrm * L, axis=1) / dist, 0, 1)
            rad += lv * (0.8 * ndl * (170.0 / dist) ** 2)[:, None] * np.array([1.0, 1.0, 1.0])
    hal = lights.get("halogen", 0.0)
    if hal > 0:
        L = _obj(S.HALOGEN_POSITION_CAD) - pts
        ndl = np.clip(np.sum(nrm * _norm(L), axis=1), 0, 1)
        rad += hal * (1.4 * ndl)[:, None] * np.array([1.0, 0.78, 0.45])
    return rad


def render(cam: str, scene: dict, size=(640, 480), exposure_us: int = 20000, gain: int = 0,
           focus: int = BEST_FOCUS, seed: int = 0) -> np.ndarray:
    """BGR uint8 image of camera `cam` ("A"/"B") for a simulator scene()."""
    import cv2
    w, h = size
    g = S.CAMERA_GEOMETRY[cam]
    pos = np.array(g["position_mm"], float)
    f, r, d = camera_basis(pos, g["aim_mm"])
    fx = (w / 2) / math.tan(math.radians(g["hfov_deg"]) / 2)
    fy = fx
    lights = scene.get("lights", {})
    img = np.zeros((h, w, 3))

    # platter: ray / plane z = 0
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = f + ((u - w / 2) / fx)[..., None] * r + ((v - h / 2) / fy)[..., None] * d
    t = -pos[2] / np.minimum(rays[..., 2], -1e-6)
    hit = pos + t[..., None] * rays
    rr = np.hypot(hit[..., 0], hit[..., 1])
    on_platter = (rays[..., 2] < 0) & (rr < S.PLATTER_RADIUS_MM)
    ang = np.degrees(np.arctan2(hit[..., 1], hit[..., 0])) - scene.get("platter_deg", 0.0)
    ang = np.mod(ang, 360.0)
    dots = np.zeros_like(on_platter)
    for k, a0 in enumerate(MARKERS_DEG):                              # irregular stickers on the platter
        dots |= on_platter & (np.abs(rr - 80) < 2.5 + (k % 3)) & (np.abs(ang - a0) < 1.5 + (k % 2))
    refl = np.where(dots, 0.5, 0.012) * on_platter
    up = np.zeros((h * w, 3))
    up[:, 2] = 1.0
    plat_rad = _light_terms(hit.reshape(-1, 3), up, lights).reshape(h, w, 3)
    img += refl[..., None] * plat_rad

    if scene.get("object_g", 0.0) > 1.0:
        pts, nrm, alb, uvm = _vessel(w).points(scene.get("platter_deg", 0.0))
        rel = pts - pos
        facing = np.sum(nrm * -rel, axis=1) > 0
        pts, nrm, alb, uvm, rel = pts[facing], nrm[facing], alb[facing], uvm[facing], rel[facing]
        zc = _dot(rel, f)
        px = np.round(w / 2 + fx * _dot(rel, r) / zc - 0.5).astype(np.int64)
        py = np.round(h / 2 + fy * _dot(rel, d) / zc - 0.5).astype(np.int64)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px, py, zc, pts, nrm, alb, uvm = px[ok], py[ok], zc[ok], pts[ok], nrm[ok], alb[ok], uvm[ok]
        col = alb * _light_terms(pts, nrm, lights)
        uvl = lights.get("uv", 0.0)
        if uvl > 0:
            # fluorescence is ~20x weaker than visible reflection: needs the long UV exposure
            col += uvl * (uvm[:, None] * np.array([0.55, 0.65, 1.0]) * 0.045 + alb * np.array([0.004, 0.002, 0.008]))
        idx = py * w + px
        depth = np.full(h * w, np.inf)
        np.minimum.at(depth, idx, zc)            # z-buffer
        sel = zc <= depth[idx] + 1e-9
        flat = img.reshape(-1, 3)
        flat[idx[sel]] = col[sel]
        img = flat.reshape(h, w, 3)
        mask = np.zeros(h * w, np.uint8)
        mask[idx] = 255
        holes = cv2.morphologyEx(mask.reshape(h, w), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > mask.reshape(h, w)
        if holes.any():
            blurred = cv2.blur(img.astype(np.float32), (3, 3))
            img[holes] = blurred[holes]

    gain_f = 1.0 + max(0, gain) / 16.0
    lin = img * (exposure_us / 20000.0) * gain_f
    rng = np.random.default_rng(seed)
    srgb = np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255 + 2.0 + rng.normal(0, 1.2, lin.shape)
    out = np.clip(srgb, 0, 255).astype(np.uint8)[..., ::-1].copy()   # RGB -> BGR
    blur = abs(focus - BEST_FOCUS) / 40.0 * (w / 1600.0)
    if blur > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    return out


class MockCamera:
    """Same interface as cameras.UvcCamera, backed by render()."""

    def __init__(self, name: str, scene_fn: Callable[[], dict], resolution=None):
        self.name = name
        self.scene_fn = scene_fn
        self.resolution = tuple(resolution or S.MOCK_RESOLUTION)
        self.controls = dict(S.CAMERA_CONTROLS.get(name, S.CAMERA_CONTROLS["A"]))
        self.device = "mock:%s" % name
        self.is_open = False
        self._n = 0

    def open(self) -> "MockCamera":
        self.is_open = True
        return self

    def lock(self, focus=None, exposure_us=None, gain=None, wb_k=None) -> dict:
        for k, v in (("focus", focus), ("exposure_us", exposure_us), ("gain", gain), ("wb_k", wb_k)):
            if v is not None:
                self.controls[k] = int(v)
        return {"applied": dict(self.controls), "readback": {}, "warnings": []}

    def warm_up(self, frames: int = None) -> None:
        pass

    def read(self) -> np.ndarray:
        if not self.is_open:
            raise RuntimeError("camera %s not open" % self.name)
        self._n += 1
        c = self.controls
        return render(self.name if self.name in ("A", "B") else "A", self.scene_fn(), self.resolution,
                      c["exposure_us"], c["gain"], c["focus"], seed=self._n)

    def close(self) -> None:
        self.is_open = False
