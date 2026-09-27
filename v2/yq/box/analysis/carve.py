"""Visual hull by space carving with sub-voxel accuracy.

For every calibrated view we turn the silhouette into a signed distance map (pixels, + inside)
and, for each voxel centre X, convert the value at its projection to millimetres (d * z / f).
F(X) = min over views (and the platter plane z) is >= 0 exactly inside the visual hull, and it
varies smoothly across the hull border, so marching cubes on F places the surface with a
fraction of a voxel of error instead of the usual +-1 voxel staircase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import MAX_OBJECT_HEIGHT_MM, PLATTER_RADIUS_MM, Camera
from .masks import signed_distance


@dataclass
class Hull:
    F: np.ndarray             # (nx, ny, nz) float32, mm, >= 0 inside
    origin: np.ndarray        # (3,) coordinates of voxel (0,0,0) centre
    voxel: float
    n_views: int


def _sample(sd: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup of sd at pixel coordinates uv (N,2); outside the image -> +inf (no information)."""
    import cv2
    n = len(uv)
    cols = 16384
    rows = (n + cols - 1) // cols
    pad = rows * cols - n
    mx = np.concatenate([uv[:, 0], np.full(pad, -10.0)]).astype(np.float32).reshape(rows, cols)
    my = np.concatenate([uv[:, 1], np.full(pad, -10.0)]).astype(np.float32).reshape(rows, cols)
    out = cv2.remap(sd, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.inf).ravel()[:n]
    h, w = sd.shape
    outside = (uv[:, 0] < -0.5) | (uv[:, 1] < -0.5) | (uv[:, 0] > w - 0.5) | (uv[:, 1] > h - 0.5)
    out[outside] = np.inf
    return out


def evaluate(points: np.ndarray, views: list, chunk: int = 2_000_000) -> np.ndarray:
    """F at (N,3) points. views: [(Camera, sd_map)]."""
    F = np.full(len(points), np.inf, dtype=np.float32)
    for s in range(0, len(points), chunk):
        P = points[s:s + chunk]
        f = np.minimum(np.full(len(P), np.inf, np.float32), P[:, 2].astype(np.float32))   # platter plane
        for cam, sd in views:
            uv, z = cam.project(P, return_depth=True)
            v = _sample(sd, uv) * (z / cam.focal)
            f = np.minimum(f, v.astype(np.float32))
        F[s:s + chunk] = f
    return F


def _grid(lo, hi, voxel):
    xs = [np.arange(lo[i], hi[i] + voxel * 0.5, voxel) for i in range(3)]
    return xs


def carve(views: list, voxel: float = 1.0, coarse: float = 4.0, bounds: Optional[tuple] = None,
          on_progress=None) -> Hull:
    """views: [(Camera at its platter angle, boolean mask[, sub-pixel border offset map])]. Returns the hull at `voxel` mm."""
    sviews = [(v[0], signed_distance(v[1], v[2] if len(v) > 2 else None)) for v in views]
    if len(sviews) < 4:
        raise ValueError("se necesitan al menos 4 vistas con silueta para el casco visual")
    if bounds is None:
        r = PLATTER_RADIUS_MM
        bounds = (np.array([-r, -r, -coarse]), np.array([r, r, MAX_OBJECT_HEIGHT_MM + 20.0]))
    lo, hi = (np.asarray(b, float) for b in bounds)
    xs = _grid(lo, hi, coarse)
    P = np.stack(np.meshgrid(*xs, indexing="ij"), axis=-1).reshape(-1, 3)
    Fc = evaluate(P, sviews)
    occ = Fc > -coarse * 0.9
    if not occ.any():
        raise ValueError("el casco visual está vacío: las siluetas no coinciden (¿calibración incorrecta?)")
    if on_progress:
        on_progress(0.3)
    pts = P[occ]
    lo2 = np.maximum(pts.min(axis=0) - 2 * coarse, lo)
    hi2 = np.minimum(pts.max(axis=0) + 2 * coarse, hi)
    lo2[2] = min(lo2[2], -2 * voxel)
    xs = _grid(lo2, hi2, voxel)
    shape = tuple(len(x) for x in xs)
    P = np.stack(np.meshgrid(*xs, indexing="ij"), axis=-1).reshape(-1, 3)
    F = evaluate(P, sviews).reshape(shape)
    F = np.where(np.isfinite(F), F, -voxel * 3).astype(np.float32)
    # close the grid: the outermost layer is always outside
    F[0], F[-1], F[:, 0], F[:, -1], F[:, :, 0], F[:, :, -1] = (-voxel,) * 6
    if on_progress:
        on_progress(1.0)
    return Hull(F=F, origin=np.array([xs[0][0], xs[1][0], xs[2][0]]), voxel=float(voxel), n_views=len(sviews))


def hull_mesh(h: Hull) -> tuple:
    """Marching cubes of F = 0 -> (vertices (N,3) mm in the object frame, faces (M,3), normals)."""
    from skimage import measure
    F = np.clip(h.F, -5 * h.voxel, 5 * h.voxel)
    verts, faces, normals, _ = measure.marching_cubes(F, level=0.0, spacing=(h.voxel,) * 3)
    verts = verts + h.origin
    if mesh_volume(verts, faces) < 0:           # make the faces wind outward (positive volume)
        faces = faces[:, ::-1].copy()
    # F grows towards the inside, so the outward normal is minus its gradient
    g = np.gradient(F, h.voxel)
    idx = np.clip(np.round((verts - h.origin) / h.voxel).astype(int), 0, np.array(F.shape) - 1)
    n = -np.stack([gi[idx[:, 0], idx[:, 1], idx[:, 2]] for gi in g], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return verts, faces, n


def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def mesh_area(verts: np.ndarray, faces: np.ndarray) -> float:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum() / 2.0)
