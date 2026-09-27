"""Small polyline toolkit shared by the ART modules.

A stroke is a float numpy array of shape (N, 2): x to the right, y DOWN (SVG /
image convention). Units are pixels or millimetres depending on the stage.
Only numpy (and OpenCV for rendering) are needed, so this runs on the Jetson.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np

Stroke = np.ndarray


def as_stroke(pts) -> Stroke:
    a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return a


def length(s: Stroke) -> float:
    if len(s) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(s, axis=0).T)))


def total_length(strokes: Iterable[Stroke]) -> float:
    return float(sum(length(s) for s in strokes))


def is_closed(s: Stroke, tol: float = 1e-6) -> bool:
    return len(s) > 2 and float(np.hypot(*(s[0] - s[-1]))) <= tol


def bounds(strokes: Sequence[Stroke]) -> tuple:
    pts = np.concatenate([s for s in strokes if len(s)], axis=0) if strokes else np.zeros((0, 2))
    if not len(pts):
        return (0.0, 0.0, 0.0, 0.0)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def transform(strokes: Sequence[Stroke], scale: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> List[Stroke]:
    return [s * scale + np.array([dx, dy]) for s in strokes]


def rotate(strokes: Sequence[Stroke], deg: float, cx: float = 0.0, cy: float = 0.0) -> List[Stroke]:
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array([cx, cy])
    return [(s - c) @ r.T + c for s in strokes]


def resample(s: Stroke, step: float) -> Stroke:
    """Points every `step` along the polyline (keeps both ends)."""
    if len(s) < 2:
        return s.copy()
    seg = np.hypot(*np.diff(s, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return s[:1].copy()
    n = max(2, int(math.ceil(total / step)) + 1)
    t = np.linspace(0.0, total, n)
    return np.stack([np.interp(t, cum, s[:, 0]), np.interp(t, cum, s[:, 1])], axis=1)


def rdp(s: Stroke, eps: float) -> Stroke:
    """Ramer-Douglas-Peucker simplification (iterative, numpy)."""
    n = len(s)
    if n < 3 or eps <= 0:
        return s.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    closed = is_closed(s)
    stack = [(0, n - 1)]
    if closed:
        # split a closed loop at its farthest point so the chord is not degenerate
        far = int(np.argmax(np.hypot(*(s - s[0]).T)))
        if 0 < far < n - 1:
            keep[far] = True
            stack = [(0, far), (far, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = s[i], s[j]
        seg = s[i + 1:j]
        ab = b - a
        L = math.hypot(ab[0], ab[1])
        if L < 1e-12:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return s[keep]


def chaikin(s: Stroke, iterations: int = 1, keep: Sequence[int] = ()) -> Stroke:
    """Chaikin corner cutting; endpoints (and indices in `keep`, e.g. sharp corners) stay put."""
    out = s
    keep_mask = np.zeros(len(s), dtype=bool)
    for k in keep:
        if 0 <= k < len(s):
            keep_mask[k] = True
    for _ in range(iterations):
        if len(out) < 3:
            return out
        closed = is_closed(out)
        pts = [out[0]]
        new_keep = [True]
        for i in range(len(out) - 1):
            p, q = out[i], out[i + 1]
            if i > 0 or closed:
                if keep_mask[i]:
                    pts.append(p)
                    new_keep.append(True)
                pts.append(0.75 * p + 0.25 * q)
                new_keep.append(False)
            if i < len(out) - 2 or closed:
                pts.append(0.25 * p + 0.75 * q)
                new_keep.append(False)
        pts.append(out[-1])
        new_keep.append(True)
        out = np.asarray(pts)
        keep_mask = np.asarray(new_keep)
    return out


def sharp_corners(s: Stroke, angle_deg: float = 55.0, reach: float = 4.0) -> List[int]:
    """Indices where the direction turns more than angle_deg, measured `reach` units away."""
    n = len(s)
    if n < 3:
        return []
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(s, axis=0).T))])
    out = []
    cos_lim = math.cos(math.radians(angle_deg))
    for i in range(1, n - 1):
        a = np.searchsorted(cum, cum[i] - reach)
        b = np.searchsorted(cum, cum[i] + reach)
        a = min(max(a, 0), i - 1)
        b = max(min(b, n - 1), i + 1)
        u = s[i] - s[a]
        v = s[b] - s[i]
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu < 1e-9 or nv < 1e-9:
            continue
        if float(np.dot(u, v)) / (nu * nv) < cos_lim:
            out.append(i)
    # keep only local maxima of turning (one index per corner)
    if not out:
        return out
    groups, cur = [], [out[0]]
    for i in out[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [g[len(g) // 2] for g in groups]


def smooth(s: Stroke, sigma: float, pinned: Sequence[int] = ()) -> Stroke:
    """Gaussian smoothing along the arc (in index space, points ~1 unit apart).
    Endpoints of open strokes and `pinned` indices stay fixed; closed loops wrap around."""
    n = len(s)
    if n < 4 or sigma <= 0:
        return s.copy()
    closed = is_closed(s)
    r = max(1, min(int(math.ceil(3 * sigma)), n - 2))        # short strokes: shorter kernel
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    if closed:
        core = s[:-1]
        pad = np.concatenate([core[-r:], core, core[:r]], axis=0) if len(core) > r else None
        if pad is None:
            return s.copy()
        sm = np.stack([np.convolve(pad[:, d], k, mode="valid") for d in (0, 1)], axis=1)
        out = np.concatenate([sm, sm[:1]], axis=0)
    else:
        # reflect around the endpoints (odd reflection keeps straight ends straight)
        head = 2 * s[0] - s[1:r + 1][::-1]
        tail = 2 * s[-1] - s[-r - 1:-1][::-1]
        pad = np.concatenate([head, s, tail], axis=0)
        out = np.stack([np.convolve(pad[:, d], k, mode="valid") for d in (0, 1)], axis=1)
        out[0], out[-1] = s[0], s[-1]
    # pinned points (sharp corners): blend back to the original around them
    for i in pinned:
        lo, hi = max(0, i - r), min(n, i + r + 1)
        for j in range(lo, hi):
            w = 1.0 - abs(j - i) / float(r + 1)
            out[j] = w * s[j] + (1 - w) * out[j]
    return out


def point_segment_dist(p: np.ndarray, strokes: Sequence[Stroke]) -> np.ndarray:
    """Distance from each point in p (M,2) to the nearest segment of `strokes` (brute force, chunked)."""
    segs = []
    for s in strokes:
        if len(s) >= 2:
            segs.append(np.concatenate([s[:-1], s[1:]], axis=1))
        elif len(s) == 1:
            segs.append(np.concatenate([s, s], axis=1))
    if not segs:
        return np.full(len(p), np.inf)
    S = np.concatenate(segs, axis=0)
    a, b = S[:, :2], S[:, 2:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    out = np.empty(len(p))
    chunk = max(1, 2_000_000 // max(1, len(S)))
    for i in range(0, len(p), chunk):
        q = p[i:i + chunk][:, None, :]
        t = np.clip(((q - a) * ab).sum(2) / L2, 0.0, 1.0)
        proj = a + t[..., None] * ab
        out[i:i + chunk] = np.sqrt(((q - proj) ** 2).sum(2)).min(1)
    return out


def hausdorff(A: Sequence[Stroke], B: Sequence[Stroke], step: float = 0.5) -> float:
    """Symmetric Hausdorff distance between two sets of polylines (sampled every `step`)."""
    pa = np.concatenate([resample(s, step) for s in A if len(s)], axis=0) if A else np.zeros((0, 2))
    pb = np.concatenate([resample(s, step) for s in B if len(s)], axis=0) if B else np.zeros((0, 2))
    if not len(pa) or not len(pb):
        return float("inf")
    return float(max(point_segment_dist(pa, B).max(), point_segment_dist(pb, A).max()))


def render(strokes: Sequence[Stroke], w: int, h: int, width_px: float = 1.0, scale: float = 1.0,
           dx: float = 0.0, dy: float = 0.0, canvas: np.ndarray = None, antialias: bool = True) -> np.ndarray:
    """Draw strokes (black on white) into a uint8 image; coordinates are multiplied by `scale`."""
    import cv2
    img = canvas if canvas is not None else np.full((h, w), 255, np.uint8)
    shift = 4
    th = max(1, int(round(width_px)))
    line = cv2.LINE_AA if antialias else cv2.LINE_8
    for s in strokes:
        if not len(s):
            continue
        p = np.round((s * scale + np.array([dx, dy])) * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        if len(s) == 1:
            c = tuple(int(v) for v in p[0, 0])
            cv2.circle(img, c, max(1, th // 2) << shift, 0, -1, line, shift)
        else:
            cv2.polylines(img, [p], False, 0, th, line, shift)
    return img


def travel(strokes: Sequence[Stroke], start=(0.0, 0.0), end=None) -> float:
    """Pen-up distance to draw `strokes` in this order from `start` (and back to `end` if given)."""
    cur = np.asarray(start, dtype=float)
    d = 0.0
    for s in strokes:
        if not len(s):
            continue
        d += float(np.hypot(*(s[0] - cur)))
        cur = s[-1]
    if end is not None:
        d += float(np.hypot(*(np.asarray(end, dtype=float) - cur)))
    return d


def clip_to_box(strokes: Sequence[Stroke], x0: float, y0: float, x1: float, y1: float) -> List[Stroke]:
    """Cut strokes at an axis-aligned box (parts outside are removed, strokes split)."""
    out: List[Stroke] = []
    for s in strokes:
        inside = (s[:, 0] >= x0) & (s[:, 0] <= x1) & (s[:, 1] >= y0) & (s[:, 1] <= y1)
        if inside.all():
            out.append(s)
            continue
        cur: list = []
        for i in range(len(s)):
            if inside[i]:
                if not cur and i > 0 and not inside[i - 1]:
                    cur.append(_clip_point(s[i - 1], s[i], x0, y0, x1, y1))
                cur.append(s[i])
            else:
                if cur:
                    cur.append(_clip_point(s[i], s[i - 1], x0, y0, x1, y1))
                    if len(cur) >= 2:
                        out.append(np.asarray(cur))
                    cur = []
        if len(cur) >= 2:
            out.append(np.asarray(cur))
    return out


def _clip_point(outside: np.ndarray, inside: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    """Point where the segment inside->outside leaves the box."""
    d = outside - inside
    t = 1.0
    for k, lo, hi in ((0, x0, x1), (1, y0, y1)):
        if d[k] > 1e-12:
            t = min(t, (hi - inside[k]) / d[k])
        elif d[k] < -1e-12:
            t = min(t, (lo - inside[k]) / d[k])
    return inside + max(0.0, min(1.0, t)) * d
