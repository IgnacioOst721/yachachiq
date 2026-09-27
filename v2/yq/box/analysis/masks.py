"""Object silhouettes on the black box interior.

Background subtraction against the empty-platter photo (same camera, same settings) when the
scan has one, else a brightness threshold against the black interior; plus soft-shadow
rejection, morphology, hole filling and a region of interest = projection of the working
cylinder (the object can only be there). No learned model: on a controlled black box this
is exact enough, fully offline and license-free (a learned segmenter was not added because it
did not measurably help on the synthetic scans and adds a model download).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import MAX_OBJECT_HEIGHT_MM, PLATTER_RADIUS_MM, Camera


def roi_mask(cam: Camera, radius: float = PLATTER_RADIUS_MM + 2.0, height: float = MAX_OBJECT_HEIGHT_MM + 15.0) -> np.ndarray:
    """Pixels where the working cylinder (r <= radius, 0 <= z <= height) projects."""
    import cv2
    a = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    ring = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)
    pts = np.concatenate([np.column_stack([ring, np.full(len(a), z)]) for z in np.linspace(-2.0, height, 8)])
    uv, z = cam.project(pts, return_depth=True)
    uv = uv[z > 1]
    m = np.zeros((cam.height, cam.width), np.uint8)
    if len(uv) >= 3:
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def linear(img: np.ndarray) -> np.ndarray:
    """sRGB-encoded 8-bit values -> linear light (partial pixel coverage mixes linearly only here)."""
    x = np.asarray(img, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _otsu(values: np.ndarray) -> float:
    import cv2
    v = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
    if v.size < 16:
        return 128.0
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def fill_holes(m: np.ndarray) -> np.ndarray:
    import cv2
    h, w = m.shape
    ff = np.pad(m.astype(np.uint8), 1)
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    holes = ff[1:-1, 1:-1] == 0
    return m | holes


def object_mask(img: np.ndarray, background: Optional[np.ndarray] = None, cam: Optional[Camera] = None,
                min_diff: float = 14.0, min_bright: float = 12.0) -> tuple:
    """Binary silhouette of the object in an RGB uint8 photo. Returns (mask, info)."""
    import cv2
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    bright = f.max(axis=2) if f.ndim == 3 else f
    roi = roi_mask(cam) if cam is not None else np.ones(bright.shape, bool)
    info = {"method": "", "threshold": 0.0}
    if background is not None and background.shape == img.shape:
        b = cv2.GaussianBlur(background.astype(np.float32), (0, 0), 1.0)
        diff = np.abs(f - b).max(axis=2) if f.ndim == 3 else np.abs(f - b)
        raw = np.abs(linear(img) - linear(background))
        raw_contrast = raw.max(axis=2) if raw.ndim == 3 else raw
        thr = max(min_diff, _otsu(diff[roi]) * 0.6)
        m = diff > thr
        if f.ndim == 3:
            # soft shadows on the platter: darker, same chromaticity as the background -> not object
            r = (f + 4.0) / (b + 4.0)
            shadow = (r.max(axis=2) < 0.93) & (r.min(axis=2) > 0.2) & ((r.max(axis=2) - r.min(axis=2)) < 0.08) & \
                     (b.max(axis=2) > 18.0)
            m &= ~shadow
        info.update(method="background", threshold=thr)
    else:
        border = np.concatenate([bright[:8].ravel(), bright[-8:].ravel(), bright[:, :8].ravel(), bright[:, -8:].ravel()])
        level = float(np.median(border))
        thr = max(level + min_bright, _otsu(bright[roi]))
        m = bright > thr
        lin = linear(img)
        rawb = lin.max(axis=2) if lin.ndim == 3 else lin
        raw_contrast = np.clip(rawb - float(linear(np.array([level]))[0]), 0, None)
        info.update(method="brightness", threshold=thr)
    m &= roi
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.nonzero(areas >= max(50, 0.15 * areas.max()))[0]
        m = np.isin(lab, keep)
    else:
        m = m.astype(bool)
    m = fill_holes(m)
    ys, xs = np.nonzero(m)
    info["area_px"] = int(m.sum())
    info["offset"] = edge_offset(raw_contrast, m)
    info["touches_border"] = bool(len(xs) and (xs.min() == 0 or ys.min() == 0 or xs.max() == m.shape[1] - 1 or ys.max() == m.shape[0] - 1))
    return m, info


def edge_offset(contrast: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sub-pixel correction of the binary border from partial pixel coverage.

    contrast: per-pixel difference to the background (>= 0). For an edge pixel covered by a
    fraction a of the object, contrast ~ a * (local object contrast). An included edge pixel moves
    the true border inward by (1 - a); an excluded one outward by a. Returned map (pixels, added
    to the signed distance) is smoothed along the border and spread over a few pixels."""
    import cv2
    m = mask.astype(np.uint8)
    core = cv2.erode(m, np.ones((5, 5), np.uint8)).astype(np.float32)
    outside = (1 - cv2.dilate(m, np.ones((5, 5), np.uint8))).astype(np.float32)
    c = contrast.astype(np.float32)
    k = (15, 15)
    d_in = cv2.blur(c * core, k) / np.maximum(cv2.blur(core, k), 1e-3)
    d_out = cv2.blur(c * outside, k) / np.maximum(cv2.blur(outside, k), 1e-3)
    a = np.clip((c - d_out) / np.maximum(d_in - d_out, 1e-3), 0.0, 1.0)
    ker = np.ones((3, 3), np.uint8)
    edge_in = (m > 0) & (cv2.erode(m, ker) == 0)
    edge_out = (m == 0) & (cv2.dilate(m, ker) > 0)
    # border position = binary border - (1 - a_in) + a_out  (the true edge lies in exactly one of the two pixels)
    off = np.zeros(m.shape, np.float32)
    for sel, val in ((edge_in, a - 1.0), (edge_out, a)):
        num = cv2.GaussianBlur(np.where(sel, val, 0).astype(np.float32), (0, 0), 1.5)
        den = cv2.GaussianBlur(sel.astype(np.float32), (0, 0), 1.5)
        off += np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    return np.clip(off, -1.0, 1.0).astype(np.float32)


def signed_distance(mask: np.ndarray, offset: Optional[np.ndarray] = None) -> np.ndarray:
    """Pixels to the silhouette border: positive inside, negative outside (float32)."""
    import cv2
    m = mask.astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    sd = (inside - outside + np.where(m > 0, -0.5, 0.5)).astype(np.float32)
    if offset is not None:
        sd = sd + offset * (np.abs(sd) < 5.0)
    return sd
