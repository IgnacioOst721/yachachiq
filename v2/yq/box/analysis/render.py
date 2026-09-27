"""Small numpy renderer for dense meshes: point-splat z-buffers, visibility and preview images.

Marching-cubes meshes have a vertex every ~0.5-1 mm, so splatting vertices (plus triangle
centres) into a z-buffer with a 3x3 minimum filter gives hole-free depth maps without OpenGL
(the kiosk Mac/Jetson may run headless).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import Camera, look_at, unit


def surface_samples(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Vertices + triangle centroids (denser splats)."""
    return np.concatenate([verts, verts[faces].mean(axis=1)])


def zbuffer(cam: Camera, pts: np.ndarray, scale: float = 0.5, grow: int = 1, leak_tol: float = 3.0) -> np.ndarray:
    """Depth map (mm along the optical axis) at `scale` of the camera resolution; inf where empty."""
    import cv2
    w, h = max(1, int(cam.width * scale)), max(1, int(cam.height * scale))
    uv, z = cam.project(pts, return_depth=True)
    uv = uv * scale
    ok = (z > 1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < w - 0.5) & (uv[:, 1] < h - 0.5)
    ix = np.round(uv[ok, 0]).astype(np.int64).clip(0, w - 1)
    iy = np.round(uv[ok, 1]).astype(np.int64).clip(0, h - 1)
    zb = np.full(h * w, np.inf, np.float32)
    np.minimum.at(zb, iy * w + ix, z[ok].astype(np.float32))
    zb = zb.reshape(h, w)
    if grow > 0:
        # fill splat holes AND replace back-surface samples that leaked through a hole of the front
        # surface: any pixel much deeper than its neighbourhood minimum takes that minimum
        k = 2 * grow + 1
        big = np.where(np.isfinite(zb), zb, 1e9).astype(np.float32)
        eroded = cv2.erode(big, np.ones((k, k), np.uint8))
        leak = np.isfinite(zb) & (zb > eroded + leak_tol)
        zb = np.where(np.isfinite(zb) & ~leak, zb, np.where(eroded < 1e9, eroded, np.inf)).astype(np.float32)
    return zb


def visibility(cam: Camera, pts: np.ndarray, normals: Optional[np.ndarray], zb: np.ndarray, scale: float,
               tol: float = 1.5, min_cos: float = 0.15) -> np.ndarray:
    """Points seen by `cam`: in front, facing it (normal . view > min_cos), not hidden (depth <= zbuffer + tol)."""
    uv, z = cam.project(pts, return_depth=True)
    h, w = zb.shape
    ix = np.round(uv[:, 0] * scale).astype(np.int64)
    iy = np.round(uv[:, 1] * scale).astype(np.int64)
    inside = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h) & (z > 1)
    vis = np.zeros(len(pts), bool)
    zz = zb[iy[inside], ix[inside]]
    vis[inside] = z[inside] <= zz + tol
    if normals is not None:
        vdir = unit(cam.center[None] - pts)
        vis &= np.einsum("ij,ij->i", normals, vdir) > min_cos
    return vis


def preview(verts: np.ndarray, faces: np.ndarray, colors: Optional[np.ndarray], normals: np.ndarray, azimuth_deg: float,
            elevation_deg: float = 25.0, size: int = 640, distance: float = 420.0,
            light=(0.3, -0.5, 0.8)) -> np.ndarray:
    """RGB uint8 render of the mesh from a virtual camera on a sphere around the object (dark background)."""
    import cv2
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    radius = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) / 2
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    eye = center + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    R, t = look_at(eye, center)
    f = size * 0.5 * distance / (radius * 1.25)
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
    cam = Camera("preview", K, R, t, size, size)
    ss = 2
    big = Camera("preview", K * np.array([[ss], [ss], [1]]), R, t, size * ss, size * ss)
    pts = surface_samples(verts, faces)
    nrm = unit(np.concatenate([normals, normals[faces].mean(axis=1)]))
    col = None if colors is None else np.concatenate([colors, colors[faces].mean(axis=1)])
    uv, z = big.project(pts, return_depth=True)
    W = size * ss
    ok = (z > 1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W - 0.5) & (uv[:, 1] < W - 0.5)
    ix, iy = np.round(uv[ok, 0]).astype(np.int64), np.round(uv[ok, 1]).astype(np.int64)
    key = iy * W + ix
    order = np.lexsort((z[ok], key))                     # nearest first within each pixel
    key_s = key[order]
    first = np.r_[True, key_s[1:] != key_s[:-1]]
    sel = np.nonzero(ok)[0][order[first]]
    L = unit(np.asarray(light, float))
    vdir = unit(eye[None] - pts[sel])
    diff = np.clip(np.einsum("ij,j->i", nrm[sel], L), 0, 1)
    hvec = unit(L[None] + vdir)
    spec = np.clip(np.einsum("ij,ij->i", nrm[sel], hvec), 0, 1) ** 30
    base = (col[sel].astype(np.float32) / 255.0) if col is not None else np.full((len(sel), 3), 0.72, np.float32)
    shade = base * (0.35 + 0.75 * diff)[:, None] + 0.12 * spec[:, None]
    img = np.zeros((W * W, 3), np.float32)
    img[:] = (0.07, 0.07, 0.08)
    img[key_s[first]] = shade
    img = img.reshape(W, W, 3)
    covered = np.zeros(W * W, np.uint8)
    covered[key_s[first]] = 1
    covered = covered.reshape(W, W)
    # close splat gaps: fill uncovered pixels inside the silhouette with the local average
    closed = cv2.morphologyEx(covered, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    blur = cv2.blur(img * covered[..., None], (5, 5)) / np.maximum(cv2.blur(covered.astype(np.float32), (5, 5)), 1e-6)[..., None]
    img = np.where(((closed > 0) & (covered == 0))[..., None], blur, img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def splat_scale(cam: Camera, verts: np.ndarray, faces: np.ndarray, target_px: float = 0.7) -> float:
    """Largest image scale at which neighbouring mesh vertices land at most `target_px` apart."""
    f = faces[:: max(1, len(faces) // 5000)]
    uv = cam.project(verts[f.reshape(-1)]).reshape(-1, 3, 2)
    edge = float(np.median(np.linalg.norm(uv[:, 1] - uv[:, 0], axis=1)))
    return float(min(1.0, target_px / max(edge, 1e-6)))


def depth_map(cam: Camera, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Hole-free depth (mm along the optical axis) at full camera resolution; inf off the object."""
    import cv2
    s = splat_scale(cam, verts, faces)
    zb = zbuffer(cam, surface_samples(verts, faces), scale=s, grow=1)
    if s >= 0.999:
        return zb
    fin = np.isfinite(zb)
    filled = np.where(fin, zb, 0).astype(np.float32)
    up = cv2.resize(filled, (cam.width, cam.height), interpolation=cv2.INTER_LINEAR)
    w = cv2.resize(fin.astype(np.float32), (cam.width, cam.height), interpolation=cv2.INTER_LINEAR)
    return np.where(w > 0.5, up / np.maximum(w, 1e-6), np.inf).astype(np.float32)
