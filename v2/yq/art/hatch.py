"""Hatching: turn filled or grey areas into parallel pen lines (engraving style).

    hatch_mask(mask, spacing, angle)       -> strokes filling a binary region
    tone_hatch(gray, spacing)              -> cross-hatching by darkness level
    hatch_rect(x0, y0, x1, y1, spacing)    -> serpentine fill of a rectangle (QR modules)

Neighbouring hatch lines are linked into one serpentine stroke whenever the
short connector stays inside the region, so a filled area costs few pen lifts.
Coordinates follow the input: pixels for masks, millimetres for rectangles.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from .geom import Stroke


def _runs(row: np.ndarray) -> List[tuple]:
    """(start, end) index pairs of True runs in a 1-D boolean array (end inclusive)."""
    if not row.any():
        return []
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def hatch_mask(mask: np.ndarray, spacing: float, angle_deg: float = 45.0, inset: float = 0.0,
               min_len: float = 2.0, link: bool = True, step: float = 0.5, phase: float = 0.5) -> List[Stroke]:
    """Parallel lines `spacing` pixels apart covering the True pixels of `mask`.

    inset: erode the region first (keeps a wide pen inside the outline).
    Returns strokes in pixel coordinates (x right, y down)."""
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if inset > 0.5:
        r = int(round(inset))
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if not m.any() or spacing <= 0:
        return []
    h, w = m.shape
    a = math.radians(angle_deg)
    u = np.array([math.cos(a), math.sin(a)])        # along the hatch lines
    n = np.array([-math.sin(a), math.cos(a)])       # across them
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], axis=1).astype(float)
    cn = pts[:, 0] * n[0] + pts[:, 1] * n[1]          # (explicit: macOS Accelerate matmul warns spuriously)
    ct = pts[:, 0] * u[0] + pts[:, 1] * u[1]
    c0, c1 = cn.min(), cn.max()
    t0, t1 = ct.min() - 1, ct.max() + 1
    offsets = np.arange(c0 + phase * spacing, c1 + 1e-9, spacing)
    if not len(offsets):
        offsets = np.array([(c0 + c1) / 2.0])
    ts = np.arange(t0, t1 + step, step)
    lines = []
    for c in offsets:
        P = c * n + ts[:, None] * u
        xi = np.round(P[:, 0]).astype(int)
        yi = np.round(P[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        inside = np.zeros(len(ts), dtype=bool)
        inside[ok] = m[yi[ok], xi[ok]] > 0
        segs = []
        for s, e in _runs(inside):
            if (e - s) * step + step < min_len:
                continue
            segs.append((c * n + ts[s] * u, c * n + ts[e] * u))
        lines.append(segs)
    if not link:
        return [np.array([p, q]) for segs in lines for p, q in segs]
    return _serpentine(lines, m, spacing)


def _inside_path(m: np.ndarray, p: np.ndarray, q: np.ndarray) -> bool:
    h, w = m.shape
    for t in np.linspace(0, 1, 7):
        x, y = p + t * (q - p)
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h) or not m[yi, xi]:
            return False
    return True


def _serpentine(lines: list, m: np.ndarray, spacing: float) -> List[Stroke]:
    """Link segments of consecutive hatch lines into zig-zag strokes."""
    max_link = 2.2 * spacing
    done: List[list] = []
    open_chains: List[list] = []           # each chain: list of points, last point is its free end
    for segs in lines:
        new_open: List[list] = []
        used = [False] * len(segs)
        # candidate links (distance, chain index, segment index, reversed)
        cands = []
        for ci, ch in enumerate(open_chains):
            e = ch[-1]
            for si, (p, q) in enumerate(segs):
                for rev, start in ((False, p), (True, q)):
                    d = float(np.hypot(*(start - e)))
                    if d <= max_link:
                        cands.append((d, ci, si, rev))
        cands.sort(key=lambda t: t[0])
        taken = set()
        for d, ci, si, rev in cands:
            if ci in taken or used[si]:
                continue
            p, q = segs[si]
            start, end = (q, p) if rev else (p, q)
            if not _inside_path(m, open_chains[ci][-1], start):
                continue
            open_chains[ci].extend([start, end])
            taken.add(ci)
            used[si] = True
            new_open.append(open_chains[ci])
        for ci, ch in enumerate(open_chains):
            if ci not in taken:
                done.append(ch)
        for si, (p, q) in enumerate(segs):
            if not used[si]:
                # alternate direction relative to the previous line so links stay short
                new_open.append([p, q] if (len(done) + si) % 2 == 0 else [q, p])
        open_chains = new_open
    done.extend(open_chains)
    return [np.asarray(ch, dtype=float) for ch in done if len(ch) >= 2]


def tone_hatch(gray: np.ndarray, spacing: float, levels=((0.78, 45.0), (0.55, -45.0), (0.32, 0.0)),
               min_area: float = 60.0, exclude: np.ndarray = None) -> List[Stroke]:
    """Engraving-style tone: every darkness level below its threshold adds one hatch direction.

    gray: float image 0 (black) .. 1 (white). exclude: mask of pixels never hatched
    (the line work itself). Returns strokes in pixels."""
    import cv2
    g = np.asarray(gray, dtype=np.float32)
    g = cv2.GaussianBlur(g, (0, 0), max(1.0, spacing * 0.6))
    out: List[Stroke] = []
    for i, (thr, ang) in enumerate(levels):
        m = (g < thr).astype(np.uint8)
        if exclude is not None:
            m[exclude > 0] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros(num, dtype=bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
        m = keep[lab].astype(np.uint8)
        out += hatch_mask(m, spacing, ang, phase=0.5 + 0.25 * i)
    return out


def hatch_rect(x0: float, y0: float, x1: float, y1: float, spacing: float, pen: float = 0.0) -> Stroke:
    """One serpentine stroke filling the rectangle with horizontal lines `spacing` apart.

    The lines are inset by pen/2 so a pen of width `pen` stays inside the rectangle."""
    xi0, xi1 = x0 + pen / 2.0, x1 - pen / 2.0
    yi0, yi1 = y0 + pen / 2.0, y1 - pen / 2.0
    if xi1 < xi0:
        xi0 = xi1 = (x0 + x1) / 2.0
    if yi1 < yi0:
        yi0 = yi1 = (y0 + y1) / 2.0
    n = max(1, int(math.ceil((yi1 - yi0) / max(spacing, 1e-6) - 1e-9)) + 1)
    ys = np.linspace(yi0, yi1, n) if n > 1 else np.array([(yi0 + yi1) / 2.0])
    pts = []
    for k, y in enumerate(ys):
        if k % 2 == 0:
            pts += [(xi0, y), (xi1, y)]
        else:
            pts += [(xi1, y), (xi0, y)]
    return np.asarray(pts, dtype=float)
