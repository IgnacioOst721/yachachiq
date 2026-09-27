"""Line-art image -> clean centreline pen strokes (the heart of the front drawing).

    trace(image) -> Trace(strokes, width, height, info)      strokes in PIXELS, y down

Pipeline (each step is a small function so tests can check it alone):
  1. load at >= 1024 px (WORK_PX), grey
  2. ink mask: background normalisation + adaptive threshold with hysteresis,
     speck removal, pinhole filling, white-on-black detection
  3. decorative frame removal (straight borders on >= 3 sides)
  4. filled regions (much thicker than the typical line) -> outline + hatching
  5. thinning -> skeleton graph (endpoints, junction clusters, edges)
  6. spur pruning (short dead ends caused by thinning, scaled to the local width)
  7. merging edges through junctions by direction continuity -> long strokes
  8. corner-preserving Gaussian smoothing + sub-pixel Douglas-Peucker
  9. tiny-stroke removal that keeps isolated small details (eyes, dots)
Pen order and the continuous-line mode live in order.py; mm fitting in layout.py.
Ideas kept from v1 robot/vectorize.py: fills -> contours, frame dropping,
straight-through walking at junctions, spur pruning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import settings
from .geom import Stroke, length, rdp, sharp_corners, smooth


@dataclass
class Trace:
    strokes: List[Stroke]          # centre lines + outlines + hatching, pixels (x right, y down)
    width: int
    height: int
    info: dict = field(default_factory=dict)


# --- 1. loading -----------------------------------------------------------------------

def load_gray(src, work_px: int = None) -> np.ndarray:
    """Path / PIL image / numpy array -> uint8 grey image whose longest side is work_px."""
    import cv2
    work_px = int(work_px or settings.WORK_PX)
    if isinstance(src, np.ndarray):
        img = src
    elif hasattr(src, "convert"):                       # PIL
        img = np.asarray(src.convert("RGB"))[:, :, ::-1]
    else:
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError("could not read image: %s" % src)
    if img.ndim == 3 and img.shape[2] == 4:             # transparent PNG: composite on white
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    h, w = img.shape
    s = work_px / float(max(h, w))
    if abs(s - 1.0) > 1e-3:
        interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=interp)
    return img


# --- 2. ink mask -------------------------------------------------------------------------

def ink_mask(gray: np.ndarray, return_norm: bool = False):
    """Binary ink mask (uint8 0/1). Handles uneven paper, faint lines and inverted images."""
    import cv2
    g = gray.astype(np.float32)
    h, w = g.shape
    scale = max(h, w) / 1000.0
    norm = _normalize(g, scale)
    if float((norm < 0.5).mean()) > 0.5:               # white lines on black: invert
        g = 255.0 - g
        norm = _normalize(g, scale)
    norm = cv2.GaussianBlur(norm, (0, 0), 0.7 * max(1.0, scale * 0.8))
    n8 = (norm * 255).astype(np.uint8)
    otsu, _ = cv2.threshold(n8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_hi = min(max(otsu / 255.0, 0.45), 0.80)           # strong ink
    t_lo = min(t_hi + 0.12, 0.90)                       # weak ink, kept only when attached to strong
    block = int(round(41 * scale)) | 1
    adapt = cv2.adaptiveThreshold(n8, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 6)
    strong = (norm < t_hi).astype(np.uint8)
    weak = ((norm < t_lo) & (adapt > 0)).astype(np.uint8) | strong
    num, lab = cv2.connectedComponents(weak, connectivity=8)
    has_strong = np.zeros(num, dtype=bool)
    has_strong[np.unique(lab[strong > 0])] = True
    has_strong[0] = False
    mask = has_strong[lab].astype(np.uint8)
    mask = remove_specks(mask, max(4, int(round(10 * scale * scale))))
    mask = fill_pinholes(mask, max(3, int(round(6 * scale * scale))))
    if return_norm:
        return mask, norm
    return mask


def _normalize(g: np.ndarray, scale: float) -> np.ndarray:
    """Divide by the paper brightness so uneven light or grey paper does not look like ink.
    The paper level comes from a heavy closing on a 1/8 image (so solid black shapes up to
    ~1/5 of the picture are not mistaken for paper) and never drops below 85 % of the
    bright-paper level."""
    import cv2
    h, w = g.shape
    small = cv2.resize(g, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(min(small.shape) / 5.0)) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3.0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    paper = float(np.percentile(g, 90))
    bg = np.maximum(bg, 0.85 * paper)
    return np.clip(g / np.maximum(bg, 1.0), 0.0, 1.0)


def remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    import cv2
    num, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(num, dtype=bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab].astype(np.uint8)


def fill_pinholes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Fill tiny white holes inside ink (they would become skeleton loops)."""
    import cv2
    inv = (1 - mask).astype(np.uint8)
    num, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    small = np.zeros(num, dtype=bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] <= max_area
    out = mask.copy()
    out[small[lab]] = 1
    return out


# --- 3. frames ----------------------------------------------------------------------------

def remove_frame(mask: np.ndarray, band: float = 0.12, cover: float = 0.55) -> tuple:
    """Remove a rectangular border (straight lines near >= 3 image sides). Returns (mask, removed)."""
    h, w = mask.shape
    bh, bw = max(2, int(h * band)), max(2, int(w * band))
    rows = mask.mean(axis=1)
    cols = mask.mean(axis=0)
    top = [y for y in range(bh) if rows[y] >= cover]
    bottom = [y for y in range(h - bh, h) if rows[y] >= cover]
    left = [x for x in range(bw) if cols[x] >= cover]
    right = [x for x in range(w - bw, w) if cols[x] >= cover]
    sides = sum(1 for s in (top, bottom, left, right) if s)
    if sides < 3:
        return mask, False
    out = mask.copy()
    for ys in (top, bottom):
        for y in ys:
            out[max(0, y - 1):y + 2, :] = 0
    for xs in (left, right):
        for x in xs:
            out[:, max(0, x - 1):x + 2] = 0
    return remove_specks(out, 8), True


def drop_frame_strokes(strokes: List[Stroke], w: int, h: int, cover: float = 0.85) -> List[Stroke]:
    """Stroke-level frame test (oval vignettes, broken borders), like v1 _drop_frame."""
    out = []
    edge = 0.06 * min(w, h)
    for s in strokes:
        x0, y0 = s.min(axis=0)
        x1, y1 = s.max(axis=0)
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        if bw >= cover and bh >= cover:
            d = np.minimum.reduce([s[:, 0], w - s[:, 0], s[:, 1], h - s[:, 1]])
            if (d < 0.14 * min(w, h)).mean() > 0.8:      # hugs the border all around
                continue
        near = np.minimum.reduce([s[:, 0], w - s[:, 0], s[:, 1], h - s[:, 1]]) < edge
        if near.all() and (bw >= 0.5 or bh >= 0.5):
            continue
        out.append(s)
    return out


# --- 4. filled regions -------------------------------------------------------------------

def typical_halfwidth(mask: np.ndarray, dist: np.ndarray = None, skel: np.ndarray = None) -> float:
    import cv2
    from .skeleton import thin
    if dist is None:
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if skel is None:
        skel = thin(mask)
    v = dist[skel > 0]
    return float(np.median(v)) if len(v) else 1.0


def split_fills(mask: np.ndarray, dist: np.ndarray, halfwidth: float, min_thick_px: float = 6.0):
    """(thin_mask, fill_mask): regions much thicker than the typical line are fills."""
    import cv2
    T = max(min_thick_px, 2.6 * halfwidth + 1.5)
    core = (dist > T).astype(np.uint8)
    if not core.any():
        return mask, np.zeros_like(mask)
    r = int(math.ceil(T)) + 1
    fill = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) & mask
    fill = remove_specks(fill, int(math.pi * (2 * T) ** 2))
    if not fill.any():
        return mask, fill
    thin_m = mask & (1 - cv2.dilate(fill, np.ones((3, 3), np.uint8)))
    return thin_m.astype(np.uint8), fill.astype(np.uint8)


def fill_strokes(fill: np.ndarray, px_per_mm: float, style: str = None, norm: np.ndarray = None) -> List[Stroke]:
    """Outline every filled region and hatch it like an engraving: the darker the region
    (from `norm`, 0 = black .. 1 = paper), the denser the lines (cross-hatching when very dark)."""
    import cv2
    from .hatch import hatch_mask
    style = style or settings.FILL_STYLE
    out: List[Stroke] = []
    contours, _ = cv2.findContours(fill.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    pen_px = settings.PEN_WIDTH_MM * px_per_mm
    for c in contours:
        if len(c) < 8:
            continue
        pts = c.reshape(-1, 2).astype(float)
        pts = np.vstack([pts, pts[:1]])
        pts = smooth(pts, 1.0)
        out.append(rdp(pts, settings.SIMPLIFY_PX))
    if style not in ("hatch", "solid"):
        return out
    base = (settings.HATCH_SPACING_MM if style == "hatch" else settings.PEN_WIDTH_MM * 0.85) * px_per_mm
    num, lab, st, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), 8)
    for k in range(1, num):
        x, y, w, h = st[k, :4]
        region = (lab[y:y + h, x:x + w] == k).astype(np.uint8)
        dark = 1.0 - float(norm[y:y + h, x:x + w][region > 0].mean()) if norm is not None else 1.0
        if style == "solid":
            passes = [(settings.HATCH_ANGLE_DEG, base)]
        elif dark > 0.75:                                   # black: hatch + cross-hatch
            passes = [(settings.HATCH_ANGLE_DEG, base), (settings.HATCH_ANGLE_DEG - 90.0, base * 2.0)]
        elif dark > 0.45:
            passes = [(settings.HATCH_ANGLE_DEG, base)]
        else:                                               # light grey: sparse lines
            passes = [(settings.HATCH_ANGLE_DEG, base * 1.9)]
        for ang, sp in passes:
            for s_ in hatch_mask(region, max(2.0, sp), ang, inset=pen_px * 0.5):
                out.append(s_ + np.array([x, y], dtype=float))
    return out


# --- 5-9. trace ------------------------------------------------------------------------------

def _line_fit(pts: np.ndarray):
    """(point, unit direction) of the least-squares line through pts."""
    c = pts.mean(axis=0)
    u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def sharpen_corners(s: np.ndarray, corners: list, ink: np.ndarray, hw: float) -> np.ndarray:
    """Thinning cuts sharp corners (the tip of an A, the V of an N): move each corner point to the
    intersection of the straight parts on both sides when that point is still on the ink."""
    if not corners:
        return s
    s = s.copy()
    h, w = ink.shape
    reach = max(3.0, 1.5 * hw)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(s, axis=0).T))])
    for i in corners:
        a0, a1 = np.searchsorted(cum, [cum[i] - 4 * reach, cum[i] - reach])
        b0, b1 = np.searchsorted(cum, [cum[i] + reach, cum[i] + 4 * reach])
        A, B = s[max(0, a0):max(0, a1)], s[min(len(s), b0):min(len(s), b1 + 1)]
        if len(A) < 2 or len(B) < 2:
            continue
        pa, da = _line_fit(A)
        pb, db = _line_fit(B)
        det = da[0] * (-db[1]) - da[1] * (-db[0])
        if abs(det) < 0.2:                                   # nearly parallel: nothing to sharpen
            continue
        t = ((pb[0] - pa[0]) * (-db[1]) - (pb[1] - pa[1]) * (-db[0])) / det
        X = pa + t * da
        if np.hypot(*(X - s[i])) > 5.0 * hw + 2.0:
            continue
        xi, yi = int(round(X[0])), int(round(X[1]))
        if 0 <= xi < w and 0 <= yi < h and ink[yi, xi]:
            s[i] = X
    return s


def _finish_stroke(s: np.ndarray, hw: float, sigma: float, eps: float, ink: np.ndarray = None) -> np.ndarray:
    if len(s) < 3:
        return s
    corners = sharp_corners(s, 50.0, reach=max(3.0, 1.5 * hw))
    if ink is not None:
        s = sharpen_corners(s, corners, ink, hw)
    s = smooth(s, sigma, pinned=corners)
    return rdp(s, eps)


def trace(src, work_px: int = None, px_per_mm: float = None, drop_frame: bool = None,
          fill_style: str = None, tones: bool = None) -> Trace:
    """Vectorize a line-art image. px_per_mm: how many working pixels make one paper mm
    (used for the pen width, hatching and the minimum stroke length); defaults to fitting
    the printable area of the configured paper."""
    import cv2
    from .skeleton import (build_graph, contract_short_edges, extend_free_ends, free_ends,
                           merge_through_junctions, prune_spurs, thin)
    info: dict = {}
    gray = load_gray(src, work_px)
    h, w = gray.shape
    if px_per_mm is None:
        x0, y0, x1, y1 = settings.printable_box()
        px_per_mm = 1.0 / min((x1 - x0) / w, (y1 - y0) / h)
    mask, norm = ink_mask(gray, return_norm=True)
    info["ink_fraction"] = round(float(mask.mean()), 4)
    drop_frame = settings.DROP_FRAME if drop_frame is None else drop_frame
    if drop_frame:
        mask, info["frame_removed"] = remove_frame(mask)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skel0 = thin(mask)
    hw = typical_halfwidth(mask, dist, skel0)
    info["halfwidth_px"] = round(hw, 2)
    thin_m, fill = split_fills(mask, dist, hw)
    info["fill_fraction"] = round(float(fill.mean()), 4)
    skel = skel0 if not fill.any() else thin(thin_m)
    g = build_graph(skel)
    info["spurs"] = prune_spurs(g, dist)
    info["crossings"] = contract_short_edges(g, dist)
    raw = merge_through_junctions(g, reach=max(6.0, 3.0 * hw), hw=hw)
    raw = extend_free_ends(raw, free_ends(g), thin_m, dist, reach=max(6.0, 3.0 * hw))
    eps = settings.SIMPLIFY_PX
    sigma = settings.SMOOTH_SIGMA_PX
    ink = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    strokes = [_finish_stroke(s, hw, sigma, eps, ink) for s in raw]
    strokes = _drop_tiny(strokes, mask, skel, px_per_mm)
    if fill.any():
        strokes += fill_strokes(fill, px_per_mm, fill_style, norm)
    if tones if tones is not None else settings.TONES:
        from .hatch import tone_hatch
        excl = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        strokes += tone_hatch(norm, settings.HATCH_SPACING_MM * px_per_mm, exclude=excl)
    if drop_frame:
        strokes = drop_frame_strokes(strokes, w, h)
    info["strokes"] = len(strokes)
    return Trace(strokes=strokes, width=w, height=h, info=info)


def _drop_tiny(strokes: List[Stroke], mask: np.ndarray, skel: np.ndarray, px_per_mm: float) -> List[Stroke]:
    """Drop leftovers shorter than MIN_STROKE_MM, but keep isolated small details: a
    component whose strokes are all short (an eye, a seed, a star dot) is kept whole,
    a tiny one as a dot."""
    import cv2
    min_len = settings.MIN_STROKE_MM * px_per_mm
    num, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    h, w = mask.shape
    comp_len: dict = {}
    comp_of = []
    for s in strokes:
        c = s[len(s) // 2] if len(s) else np.zeros(2)
        xi, yi = min(w - 1, max(0, int(round(c[0])))), min(h - 1, max(0, int(round(c[1]))))
        k = int(lab[yi, xi])
        if k == 0:                                      # centroid of a junction snap may fall outside
            k = int(lab[min(h - 1, max(0, int(round(s[0, 1])))), min(w - 1, max(0, int(round(s[0, 0]))))])
        comp_of.append(k)
        comp_len[k] = comp_len.get(k, 0.0) + length(s)
    out = []
    for s, k in zip(strokes, comp_of):
        L = length(s)
        if L >= min_len:
            out.append(s)
        elif comp_len.get(k, 0.0) < min_len and k != 0:  # the whole component is a small detail
            out.append(s if L > 0.5 else s[:1])
    # components that produced no stroke at all (round dots thinned to one pixel)
    seen = set(comp_of)
    ys, xs = np.nonzero(skel)
    if len(ys):
        ks = lab[ys, xs]
        for k in np.unique(ks):
            if k and k not in seen:
                i = int(np.flatnonzero(ks == k)[0])
                out.append(np.array([[float(xs[i]), float(ys[i])]]))
    return out
