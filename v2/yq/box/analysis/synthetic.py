"""Synthetic ground truth: analytic objects on the platter rendered by exact ray casting.

Used by the tests (and `tools/box_analysis_demo.py`) to prove the analyses before the box
exists: we know every dimension, volume and normal of these shapes exactly.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .geometry import PLATTER_RADIUS_MM, Camera, unit

INF = np.inf


class Shape:
    """Solid resting on the platter (z >= 0) in the object frame."""
    name = "shape"

    def intersect(self, o, d) -> np.ndarray:           # (N,) distance along unit d, inf if missed
        raise NotImplementedError

    def normal(self, p) -> np.ndarray:
        raise NotImplementedError

    def volume(self) -> float:
        raise NotImplementedError

    def dims(self) -> dict:                             # height, width (longest horizontal), depth
        raise NotImplementedError

    def uv(self, p) -> np.ndarray:                      # texture coordinates in mm-ish units
        ang = np.arctan2(p[:, 1], p[:, 0])
        return np.stack([ang * 40.0, p[:, 2]], axis=1)

    def albedo(self, p) -> np.ndarray:
        """Textured, colourful surface (photogrammetry needs texture): stripes + checker + noise."""
        uv = self.uv(p)
        chk = ((np.floor(uv[:, 0] / 9.0) + np.floor(uv[:, 1] / 9.0)) % 2)
        stripe = 0.5 + 0.5 * np.sin(uv[:, 0] * 0.9 + uv[:, 1] * 0.35)
        base = np.array([0.72, 0.46, 0.30])            # terracotta
        dark = np.array([0.25, 0.14, 0.10])
        cream = np.array([0.85, 0.80, 0.68])
        c = base[None] * (1 - 0.5 * chk[:, None]) + dark[None] * 0.5 * chk[:, None]
        c = c * (0.75 + 0.25 * stripe[:, None])
        band = (np.abs(((uv[:, 1] + 3) % 30.0) - 15.0) < 2.0)[:, None]
        return np.where(band, cream[None], c)


class Cylinder(Shape):
    name = "cylinder"

    def __init__(self, radius: float, height: float, cx: float = 0.0, cy: float = 0.0):
        self.r, self.h, self.c = float(radius), float(height), np.array([cx, cy])

    def intersect(self, o, d):
        ox, oy = o[..., 0] - self.c[0], o[..., 1] - self.c[1]
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (ox * d[:, 0] + oy * d[:, 1])
        cc = ox ** 2 + oy ** 2 - self.r ** 2
        disc = b * b - 4 * a * cc
        t = np.full(len(d), INF)
        ok = (disc >= 0) & (a > 1e-12)
        sq = np.sqrt(np.where(ok, disc, 0))
        t1 = np.where(ok, (-b - sq) / np.where(a > 1e-12, 2 * a, 1), INF)
        z1 = o[..., 2] + t1 * d[:, 2]
        side = ok & (t1 > 0) & (z1 >= 0) & (z1 <= self.h)
        t = np.where(side, t1, t)
        tc = (self.h - o[..., 2]) / np.where(np.abs(d[:, 2]) > 1e-12, d[:, 2], 1e-12)
        pc = o[None, :2] + tc[:, None] * d[:, :2] if np.ndim(o) == 1 else o[:, :2] + tc[:, None] * d[:, :2]
        cap = (tc > 0) & (np.sum((pc - self.c) ** 2, axis=1) <= self.r ** 2)
        return np.where(cap & (tc < t), tc, t)

    def normal(self, p):
        n = np.zeros_like(p)
        top = p[:, 2] >= self.h - 1e-6
        n[:, :2] = p[:, :2] - self.c
        n[top] = [0, 0, 1]
        return unit(n)

    def volume(self):
        return math.pi * self.r ** 2 * self.h

    def dims(self):
        return {"height": self.h, "width": 2 * self.r, "depth": 2 * self.r}


class Box(Shape):
    name = "box"

    def __init__(self, sx: float, sy: float, sz: float, yaw_deg: float = 0.0):
        self.s = np.array([sx, sy, sz], dtype=float)
        a = math.radians(yaw_deg)
        self.Rl = np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1]])  # world->local

    def _local(self, o, d):
        return (np.atleast_2d(o) @ self.Rl.T) - [0, 0, self.s[2] / 2], d @ self.Rl.T

    def intersect(self, o, d):
        ol, dl = self._local(o, d)
        half = self.s / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / np.where(np.abs(dl) < 1e-12, 1e-12, dl)
            t1, t2 = (-half - ol) * inv, (half - ol) * inv
        tmin = np.max(np.minimum(t1, t2), axis=1)
        tmax = np.min(np.maximum(t1, t2), axis=1)
        hit = (tmax >= tmin) & (tmin > 0)
        return np.where(hit, tmin, INF)

    def normal(self, p):
        pl = (p @ self.Rl.T) - [0, 0, self.s[2] / 2]
        k = np.argmax(np.abs(pl) / (self.s / 2), axis=1)
        nl = np.zeros_like(pl)
        nl[np.arange(len(pl)), k] = np.sign(pl[np.arange(len(pl)), k])
        return nl @ self.Rl

    def uv(self, p):
        pl = p @ self.Rl.T
        return np.stack([pl[:, 0] + pl[:, 1], pl[:, 2] + 0.5 * pl[:, 0]], axis=1)

    def volume(self):
        return float(np.prod(self.s))

    def dims(self):
        return {"height": self.s[2], "width": max(self.s[0], self.s[1]), "depth": min(self.s[0], self.s[1])}


class Ellipsoid(Shape):
    """Ellipsoid with semi-axes a (x), b (y), c (z), centre at height z0 < c: the platter cuts a flat base."""
    name = "ellipsoid"

    def __init__(self, a: float, b: float, c: float, z0: Optional[float] = None):
        self.ax = np.array([a, b, c], dtype=float)
        self.z0 = float(0.8 * c if z0 is None else z0)

    def intersect(self, o, d):
        oc = (np.atleast_2d(o) - [0, 0, self.z0]) / self.ax
        dd = d / self.ax
        A = np.sum(dd * dd, axis=1)
        B = 2 * np.sum(oc * dd, axis=1)
        C = np.sum(oc * oc, axis=1) - 1
        disc = B * B - 4 * A * C
        ok = disc >= 0
        t1 = np.where(ok, (-B - np.sqrt(np.where(ok, disc, 0))) / (2 * A), INF)
        z = np.atleast_2d(o)[:, 2] + t1 * d[:, 2]
        return np.where(ok & (t1 > 0) & (z >= 0), t1, INF)

    def normal(self, p):
        return unit((p - [0, 0, self.z0]) / self.ax ** 2)

    def volume(self):
        a, b, c = self.ax
        lo = -self.z0                                   # plane z=0 in centred coordinates
        return math.pi * a * b / c ** 2 * ((c ** 2 * c - c ** 3 / 3) - (c ** 2 * lo - lo ** 3 / 3))

    def dims(self):
        a, b, _ = self.ax
        return {"height": self.z0 + self.ax[2], "width": 2 * max(a, b), "depth": 2 * min(a, b)}


def _pixel_grid(w: int, h: int, ss: int) -> np.ndarray:
    off = (np.arange(ss) + 0.5) / ss - 0.5
    u = (np.arange(w)[:, None] + off[None, :]).ravel()
    v = (np.arange(h)[:, None] + off[None, :]).ravel()
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


def trace(shape: Optional[Shape], cam: Camera, ss: int = 2, chunk: int = 400000):
    """Cast one ray per sub-pixel. Returns dict of (H*ss, W*ss) arrays: t_obj, p_obj, n_obj, hit_platter, p_platter."""
    uv = _pixel_grid(cam.width, cam.height, ss)
    o, d = cam.rays(uv)
    n = len(d)
    t_obj = np.full(n, INF)
    for s in range(0, n, chunk):
        if shape is not None:
            t_obj[s:s + chunk] = shape.intersect(o, d[s:s + chunk])
    tp = -o[2] / np.where(np.abs(d[:, 2]) > 1e-12, d[:, 2], -1e-12)
    pp = o[None] + tp[:, None] * d
    platter = (tp > 0) & (np.hypot(pp[:, 0], pp[:, 1]) <= PLATTER_RADIUS_MM) & (tp < t_obj)
    hit = np.isfinite(t_obj)
    p = o[None] + np.where(hit, t_obj, 0)[:, None] * d
    return {"hit": hit, "p": p, "d": d, "o": o, "platter": platter, "pp": pp, "shape": (cam.height * ss, cam.width * ss), "ss": ss}


def shade(tr: dict, shape: Optional[Shape], lights: list, ambient: float = 0.0, platter_albedo: float = 0.05,
          wall: float = 0.02, specular: float = 0.0, shininess: float = 40.0, rgb_albedo=None) -> np.ndarray:
    """Radiance image (H, W, 3) in [0, ~1] averaged over sub-pixels.

    lights: [{"pos": (3,), "power": float, "kind": "point"|"dir", "dir": (3,)}]; point lights have 1/r^2
    falloff normalised at 150 mm. Convex shapes self-shadow exactly through n.l < 0.
    """
    hit, p = tr["hit"], tr["p"]
    out = np.zeros((len(hit), 3))
    out[:] = wall
    if tr["platter"].any():
        out[tr["platter"]] = platter_albedo * (ambient + 0.6)
    if shape is not None and hit.any():
        ph = p[hit]
        n = shape.normal(ph)
        alb = rgb_albedo(ph) if rgb_albedo is not None else shape.albedo(ph)
        v = -tr["d"][hit]
        acc = np.full(len(ph), ambient)[:, None] * alb
        for L in lights:
            if L.get("kind", "point") == "dir":
                l = np.broadcast_to(unit(np.asarray(L["dir"], float)), ph.shape)
                fall = L.get("power", 1.0)
            else:
                dv = np.asarray(L["pos"], float)[None] - ph
                r = np.linalg.norm(dv, axis=1)
                l = dv / r[:, None]
                fall = L.get("power", 1.0) * (150.0 / r) ** 2
                if "axis" in L:                                  # LED emission ~ cosine around its axis
                    ax = unit(np.asarray(L["axis"], float))
                    fall = fall * np.clip(-(l @ ax), 0.05, 1.0)
            ndl = np.clip(np.sum(n * l, axis=1), 0, None)
            acc += (fall * ndl)[:, None] * alb
            if specular > 0:
                hvec = unit(l + v)
                acc += (specular * fall * (ndl > 0) * np.clip(np.sum(n * hvec, axis=1), 0, None) ** shininess)[:, None]
        out[hit] = acc
    H, W = tr["shape"]
    ss = tr["ss"]
    img = out.reshape(H, W, 3)
    return img.reshape(H // ss, ss, W // ss, ss, 3).mean(axis=(1, 3))


def to_uint8(img: np.ndarray, gain: float = 1.0, noise: float = 0.0, rng=None) -> np.ndarray:
    x = img * gain
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        x = x + rng.normal(0, noise, x.shape)
    x = np.clip(x, 0, 1) ** (1 / 2.2)                     # sRGB-like encoding
    return (x * 255 + 0.5).astype(np.uint8)


def render_view(shape: Optional[Shape], cam: Camera, lights: Optional[list] = None, ss: int = 2, **kw) -> np.ndarray:
    """RGB uint8 photo of `shape` (None = empty platter) through `cam` (already at its platter angle)."""
    lights = lights if lights is not None else [{"kind": "dir", "dir": (0.2, -0.3, 1.0), "power": 0.55}]
    tr = trace(shape, cam, ss=ss)
    noise, rng = kw.pop("noise", 0.0), kw.pop("rng", None)
    return to_uint8(shade(tr, shape, lights, ambient=kw.pop("ambient", 0.35), **kw), noise=noise, rng=rng)
