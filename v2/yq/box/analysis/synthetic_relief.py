"""Synthetic relief tile (plaque with incised lines and a raised boss) for RTI tests.

The tile lies on the platter; its top is a height field z = h(x, y). Rays are intersected by
marching + bisection, normals come from the gradient and cast shadows are traced over the height
field, so photometric stereo is tested with realistic shadows inside the incisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import Camera, unit
from .synthetic import Shape, to_uint8


class ReliefTile(Shape):
    name = "relief_tile"

    def __init__(self, sx: float = 90.0, sy: float = 70.0, base: float = 8.0, groove_depth: float = 0.6,
                 groove_width: float = 1.4, boss: float = 2.5):
        self.sx, self.sy, self.base = sx, sy, base
        self.gd, self.gw, self.boss = groove_depth, groove_width, boss

    def h(self, x, y):
        z = np.full(np.shape(x), self.base, dtype=float)
        r2 = (x + 22.0) ** 2 + (y - 5.0) ** 2
        z = z + self.boss * np.clip(1 - r2 / 14.0 ** 2, 0, None) ** 1.5

        def groove(dist):
            return self.gd * np.clip(1 - (dist / (self.gw / 2)) ** 2, 0, None)
        for yl in (-18.0, -8.0, 2.0):
            z = z - groove(np.abs(y - yl)) * ((x > 2.0) & (x < 38.0))
        rc = np.sqrt((x - 20.0) ** 2 + (y - 20.0) ** 2)
        return z - groove(np.abs(rc - 9.0))

    def grad(self, x, y, e: float = 0.02):
        return ((self.h(x + e, y) - self.h(x - e, y)) / (2 * e), (self.h(x, y + e) - self.h(x, y - e)) / (2 * e))

    def inside_xy(self, x, y):
        return (np.abs(x) <= self.sx / 2) & (np.abs(y) <= self.sy / 2)

    def _march(self, oo, dd, ta, tb, step: float = 0.15):
        res = np.full(len(oo), np.inf)
        prev, cur = ta.copy(), ta.copy()
        found = np.zeros(len(oo), bool)
        while True:
            act = ~found & (cur < tb)
            if not act.any():
                return res
            cur = np.where(act, np.minimum(cur + step, tb), cur)
            q = oo + cur[:, None] * dd
            newly = act & (q[:, 2] - self.h(q[:, 0], q[:, 1]) <= 0)
            if newly.any():
                a, b = prev[newly], cur[newly]
                for _ in range(20):
                    m = (a + b) / 2
                    qm = oo[newly] + m[:, None] * dd[newly]
                    above = qm[:, 2] - self.h(qm[:, 0], qm[:, 1]) > 0
                    a, b = np.where(above, m, a), np.where(above, b, m)
                res[newly] = (a + b) / 2
                found |= newly
            prev = np.where(act, cur, prev)

    def intersect(self, o, d):
        o = np.broadcast_to(np.atleast_2d(o), d.shape)
        lo = np.array([-self.sx / 2, -self.sy / 2, 0.0])
        hi = np.array([self.sx / 2, self.sy / 2, self.base + self.boss + 0.5])
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
            t1, t2 = (lo - o) * inv, (hi - o) * inv
        t0 = np.max(np.minimum(t1, t2), axis=1)
        tf = np.min(np.maximum(t1, t2), axis=1)
        t = np.full(len(d), np.inf)
        idx = np.nonzero((tf >= t0) & (tf > 0))[0]
        if not len(idx):
            return t
        oo, dd, ta, tb = o[idx], d[idx], np.maximum(t0[idx], 0), tf[idx]
        p = oo + ta[:, None] * dd
        wall = p[:, 2] <= self.h(p[:, 0], p[:, 1]) + 1e-9          # entered through a side wall
        t[idx[wall]] = ta[wall]
        rem = ~wall
        t[idx[rem]] = self._march(oo[rem], dd[rem], ta[rem], tb[rem])
        return t

    def normal(self, p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        gx, gy = self.grad(x, y)
        n = unit(np.stack([-gx, -gy, np.ones_like(x)], axis=1))
        side = z < self.h(x, y) - 0.05
        ax = np.abs(x) / (self.sx / 2) >= np.abs(y) / (self.sy / 2)
        zero = 0 * x
        sn = np.where(ax[:, None], np.stack([np.sign(x), zero, zero], 1), np.stack([zero, np.sign(y), zero], 1))
        return np.where(side[:, None], sn, n)

    def shadowed(self, p, light_pos, max_dist: float = 8.0, steps: int = 24) -> np.ndarray:
        l = unit(np.asarray(light_pos, float)[None] - p)
        s = np.zeros(len(p), bool)
        for k in range(1, steps + 1):
            q = p + l * (max_dist * k / steps)
            s |= self.inside_xy(q[:, 0], q[:, 1]) & (q[:, 2] < self.h(q[:, 0], q[:, 1]) - 0.01)
        return s

    def volume(self):
        return self.sx * self.sy * self.base

    def dims(self):
        return {"height": self.base + self.boss, "width": self.sx, "depth": self.sy}

    def mesh(self, step: float = 0.5):
        """Triangulated top surface (stand-in for the reconstructed mesh in RTI tests)."""
        xs = np.arange(-self.sx / 2, self.sx / 2 + 1e-6, step)
        ys = np.arange(-self.sy / 2, self.sy / 2 + 1e-6, step)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        V = np.stack([X.ravel(), Y.ravel(), self.h(X, Y).ravel()], axis=1)
        ny = len(ys)
        i = (np.arange(len(xs) - 1)[:, None] * ny + np.arange(ny - 1)[None]).ravel()
        F = np.concatenate([np.stack([i, i + ny, i + 1], 1), np.stack([i + 1, i + ny, i + ny + 1], 1)])
        return V, F


def render_rti_tile(tile: ReliefTile, cam: Camera, lights: dict, rng=None, noise: float = 0.003):
    """Raking-light photos {index: RGB uint8, "ambient": ...} and truth maps (hit, normals, shadows)."""
    rng = rng or np.random.default_rng(0)
    H, W = cam.height, cam.width
    vv, uu = np.mgrid[0:H, 0:W]
    o, d = cam.rays(np.stack([uu.ravel(), vv.ravel()], 1).astype(float))
    t = tile.intersect(o, d)
    hit = np.isfinite(t)
    p = o[None] + np.where(hit, t, 0)[:, None] * d
    n = np.zeros_like(p)
    n[hit] = tile.normal(p[hit])
    ph = p[hit]
    alb = np.array([0.62, 0.50, 0.40]) * (0.85 + 0.15 * np.sin(ph[:, :1] * 0.4) * np.cos(ph[:, 1:2] * 0.3))
    imgs, shadows = {}, {}
    for i, lp in sorted(lights.items()):
        dv = np.asarray(lp, float)[None] - ph
        r = np.linalg.norm(dv, axis=1)
        l = dv / r[:, None]
        ndl = np.clip(np.sum(n[hit] * l, 1), 0, None)
        sh = tile.shadowed(ph, lp)
        axis = unit(np.array([0, 0, 40.0]) - np.asarray(lp, float))
        emit = np.clip(np.sum(-l * axis, 1), 0.05, 1)
        val = np.full((len(p), 3), 0.004)
        val[hit] = alb * (0.9 * emit * (150 / r) ** 2 * ndl * ~sh)[:, None]
        imgs[i] = to_uint8(val.reshape(H, W, 3), noise=noise, rng=rng)
        s = np.zeros(len(p), bool)
        s[hit] = sh
        shadows[i] = s.reshape(H, W)
    imgs["ambient"] = to_uint8(np.full((H, W, 3), 0.004), noise=noise, rng=rng)
    return imgs, {"hit": hit.reshape(H, W), "normals": n.reshape(H, W, 3), "shadows": shadows,
                  "points": p.reshape(H, W, 3)}


def write_rti_scan(folder: Path, tile: ReliefTile, cam: Camera, lights: dict, rng=None) -> dict:
    """Minimal scan folder with only rti/ (+ meta.json). Returns the truth maps."""
    import cv2
    folder = Path(folder)
    imgs, truth = render_rti_tile(tile, cam, lights, rng)
    (folder / "rti").mkdir(parents=True, exist_ok=True)
    for k, im in imgs.items():
        name = "ambient.jpg" if k == "ambient" else "led%d.jpg" % k
        cv2.imwrite(str(folder / "rti" / name), im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 97])
    lights_json = {"camera": cam.name, "platter_deg": 0.0,
                   "lights": [{"index": i, "position_mm": [float(v) for v in p]} for i, p in sorted(lights.items())]}
    (folder / "rti" / "lights.json").write_text(json.dumps(lights_json))
    (folder / "meta.json").write_text(json.dumps({"scan_id": folder.name, "profile": "standard", "analyses": ["rti"]}))
    return truth
