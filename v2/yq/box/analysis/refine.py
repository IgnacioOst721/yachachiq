"""Photo-consistency carving of the visual hull (the 'space carving' step).

Silhouettes cannot see flat tops or flat faces that no camera looks at edge-on: the visual hull
puts a 'roof' over them (a 100 mm cylinder measures ~110 mm tall from hull alone). Here every
hull surface point searches inward along its normal for the depth where small textured patches
look the same in all cameras that see it (normalised cross-correlation, so shading changes as the
object turns do not matter). Only confident, textured points move; textureless areas keep the
hull (conservative: never carves what it cannot see). Result: a refined mesh inside the hull.
"""
from __future__ import annotations

import numpy as np

from .geometry import unit
from .render import surface_samples, visibility, zbuffer


def _sample_gray(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    import cv2
    n = len(uv)
    cols = 16384
    rows = (n + cols - 1) // cols
    pad = rows * cols - n
    mx = np.concatenate([uv[:, 0], np.full(pad, -10.0)]).astype(np.float32).reshape(rows, cols)
    my = np.concatenate([uv[:, 1], np.full(pad, -10.0)]).astype(np.float32).reshape(rows, cols)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()[:n]


def _norm_patch(p: np.ndarray):
    m = p - p.mean(axis=1, keepdims=True)
    s = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(s, 1e-6), s[:, 0] / np.sqrt(p.shape[1])


def neighbors(faces: np.ndarray, n: int, width: int = 16) -> np.ndarray:
    """(n, width) 1-ring vertex neighbour indices padded with -1."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(np.concatenate([e, e[:, ::-1]]), axis=1), axis=0)
    e = np.concatenate([e, e[:, ::-1]])
    e = e[np.argsort(e[:, 0], kind="stable")]
    start = np.searchsorted(e[:, 0], np.arange(n))
    rank = np.arange(len(e)) - start[e[:, 0]]
    keep = rank < width
    out = np.full((n, width), -1, np.int64)
    out[e[keep, 0], rank[keep]] = e[keep, 1]
    return out


def mesh_median(values: np.ndarray, faces: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Median of each vertex value with its 1-ring (removes single-vertex spikes)."""
    nb = neighbors(faces, len(values))
    v = values.astype(np.float64)
    for _ in range(iterations):
        g = np.where(nb >= 0, v[np.maximum(nb, 0)], np.nan)
        v = np.nanmedian(np.concatenate([v[:, None], g], axis=1), axis=1)
    return v


def _pick_samples(verts: np.ndarray, spacing: float) -> np.ndarray:
    key = np.floor(verts / spacing).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return np.sort(idx)


def refine_surface(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray, views: list, mm_per_px: float,
                   max_depth: float = None, min_texture: float = 2.0, min_ncc: float = 0.6, min_gain: float = 0.12,
                   log=None) -> tuple:
    """views: [(Camera, gray float32 image)] at the same resolution the cameras describe.
    Returns (refined verts, info dict with per-vertex displacement and coverage stats)."""
    size = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
    max_depth = max_depth or min(30.0, 0.35 * size)
    step = max(0.3, 0.8 * mm_per_px)
    depths = np.arange(0.0, max_depth + 1e-6, step)
    spacing = max(1.5, 2.5 * mm_per_px)
    si = _pick_samples(verts, spacing)
    X0, N = verts[si], unit(normals[si])
    S, V = len(si), len(views)
    pts = surface_samples(verts, faces)
    scale = 0.5
    vis = np.zeros((S, V), bool)
    cosv = np.zeros((S, V), np.float32)
    for k, (cam, _img) in enumerate(views):
        zb = zbuffer(cam, pts, scale=scale)
        vis[:, k] = visibility(cam, X0, N, zb, scale, tol=max(1.5, 2 * mm_per_px), min_cos=0.25)
        cosv[:, k] = np.einsum("ij,ij->i", N, unit(cam.center[None] - X0))
    cosv[~vis] = -1
    ref = np.argmax(cosv, axis=1)
    usable = vis.sum(axis=1) >= 3
    # tangent patch 5x5, spacing ~1.5 px on the object
    a = unit(np.cross(N, np.where(np.abs(N[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])))
    b = np.cross(N, a)
    g = (np.arange(5) - 2) * 1.5 * mm_per_px
    gu, gv = np.meshgrid(g, g)
    offs = gu.ravel()[None, :, None] * a[:, None, :] + gv.ravel()[None, :, None] * b[:, None, :]    # (S,25,3)
    scores = np.full((S, len(depths)), -1.0, np.float32)
    texture = np.zeros(S, np.float32)
    for di, d in enumerate(depths):
        P = (X0 - d * N)[:, None, :] + offs                  # (S,25,3)
        flat = P.reshape(-1, 3)
        refp = np.zeros((S, 25), np.float32)
        for k, (cam, img) in enumerate(views):
            sel = np.nonzero(usable & (ref == k))[0]
            if len(sel):
                uv = cam.project(P[sel].reshape(-1, 3))
                refp[sel] = _sample_gray(img, uv).reshape(-1, 25)
        rn, rstd = _norm_patch(refp)
        if di == 0:
            texture = rstd
        acc = np.zeros(S, np.float32)
        cnt = np.zeros(S, np.float32)
        for k, (cam, img) in enumerate(views):
            sel = np.nonzero(usable & vis[:, k] & (ref != k))[0]
            if not len(sel):
                continue
            uv = cam.project(P[sel].reshape(-1, 3))
            pn, _ = _norm_patch(_sample_gray(img, uv).reshape(-1, 25))
            acc[sel] += np.sum(pn * rn[sel], axis=1)
            cnt[sel] += 1
        scores[:, di] = np.where(cnt >= 2, acc / np.maximum(cnt, 1), -1)
        del flat
    best = np.argmax(scores, axis=1)
    sbest = scores[np.arange(S), best]
    ok = usable & (texture >= min_texture) & (sbest >= min_ncc) & ((sbest - scores[:, 0]) >= min_gain)
    # sub-step peak: parabola through the best score and its two neighbours
    b = np.clip(best, 1, len(depths) - 2)
    s0, s1, s2 = scores[np.arange(S), b - 1], scores[np.arange(S), b], scores[np.arange(S), b + 1]
    den = s0 - 2 * s1 + s2
    frac = np.where((den < -1e-6) & (best == b), np.clip(0.5 * (s0 - s2) / np.where(den < -1e-6, den, -1), -0.5, 0.5), 0.0)
    dstar = np.where(ok, np.clip(depths[best] + frac * step, 0, None), 0.0)
    confident0 = usable & (texture >= min_texture) & (scores[:, 0] >= min_ncc) & ~ok
    # propagate to every vertex: Gaussian-weighted average of nearby decided samples
    from scipy.spatial import cKDTree
    decided = ok | confident0
    disp = np.zeros(len(verts))
    if decided.any():
        tree = cKDTree(X0[decided])
        rad = 3.0 * spacing
        dist, idx = tree.query(verts, k=8, distance_upper_bound=rad)
        w = np.where(np.isfinite(dist), np.exp(-(dist / spacing) ** 2), 0.0)
        vals = np.concatenate([dstar[decided], [0.0]])[np.minimum(idx, decided.sum())]
        # robust: weighted median would be ideal; a weighted mean after dropping outliers is enough here
        wsum = w.sum(axis=1)
        mean = (w * vals).sum(axis=1) / np.maximum(wsum, 1e-9)
        disp = np.where(wsum > 1e-6, mean, 0.0)
    disp = mesh_median(disp, faces, iterations=2)
    new = verts - disp[:, None] * unit(normals)
    up = N[:, 2] > 0.7
    info = {"samples": int(S), "usable": int(usable.sum()), "moved": int(ok.sum()), "kept": int(confident0.sum()),
            "top_coverage": float((decided & up).sum() / max(1, (usable & up).sum())),
            "textureless": int((usable & (texture < min_texture)).sum()), "max_displacement_mm": float(disp.max()) if len(disp) else 0.0,
            "mean_displacement_mm": float(disp.mean()) if len(disp) else 0.0, "coverage": float(decided.sum() / max(1, usable.sum())),
            "displacement": disp}
    if log:
        log("refine: %(samples)d samples, %(usable)d usable, %(moved)d moved, %(kept)d confirmed, %(textureless)d textureless" % info)
    return new, info
