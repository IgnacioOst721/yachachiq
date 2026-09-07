"""Image -> pen strokes, and stroke utilities.

    trace_image(path)            -> (polylines_px, w, h)   pixel space, y grows DOWN
    fit_to_paper(pls_px, w, h)   -> polylines_mm            paper space, y grows UP
    clamp(pls_mm)                -> inside [0,W]x[0,H]
    order_strokes(pls_mm)        -> greedy nearest-neighbour order (less pen travel)
    stats(pls_mm)                -> dict(strokes, points, draw_mm, travel_mm, seconds)
    to_svg(pls_mm)               -> SVG string (one <polyline> per stroke)
    to_png(pls_mm, path)         -> preview image

TRACE_MODE "lines": threshold + thinning, right for black-ink line drawings
(what we ask ComfyUI for). "edges": Canny, for photos or shaded pictures.
"""
import math

import numpy as np

try:
    import cv2
except ImportError:          # pragma: no cover
    cv2 = None

import config


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv is not installed: pip install opencv-contrib-python-headless")


# --- tracing ---------------------------------------------------------------------------------------

def _load_gray(path):
    _require_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    h, w = img.shape[:2]
    longest = max(w, h)
    if longest > config.TRACE_MAX_PX:
        s = config.TRACE_MAX_PX / float(longest)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return img


def _ink_mask(gray):
    """Binary mask of the ink (255 where the pen should draw)."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() > 127:                # more ink than paper: the picture is inverted
        mask = 255 - mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _skeleton(mask):
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # fallback: morphological skeleton (thicker/rougher than Zhang-Suen, but works anywhere)
    skel = np.zeros_like(mask)
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()
    while True:
        er = cv2.erode(img, el)
        op = cv2.dilate(er, el)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, op))
        img = er
        if cv2.countNonZero(img) == 0:
            break
    return skel


_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _prune_spurs(pts, deg, max_len):
    """Delete short dead-end branches (thinning artefacts) shorter than max_len px."""
    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in _N8 if (y + dy, x + dx) in pts]
    for _ in range(2):
        removed = False
        for e in [p for p in pts if deg.get(p) == 1]:
            if e not in pts:
                continue
            branch, cur, prev = [e], e, None
            while len(branch) <= max_len:
                nx = [q for q in nbrs(cur) if q != prev and q not in branch]
                if len(nx) != 1:
                    break
                prev, cur = cur, nx[0]
                if deg.get(cur, 0) >= 3:      # reached a junction: this was a spur
                    for b in branch:
                        pts.discard(b)
                    removed = True
                    break
                branch.append(cur)
        if not removed:
            break
        deg.clear()
        deg.update({p: len(nbrs(p)) for p in pts})
    return pts, deg


def _trace_skeleton(skel):
    """Walk a 1-px-wide skeleton into few, long polylines (pixel coords, y down)."""
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))
    if not pts:
        return []

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in _N8 if (y + dy, x + dx) in pts]

    deg = {p: len(nbrs(p)) for p in pts}
    pts, deg = _prune_spurs(pts, deg, 8)

    # connected components
    remaining_all = set(pts)
    comps = []
    while remaining_all:
        seed = remaining_all.pop()
        comp, stack = {seed}, [seed]
        while stack:
            for q in nbrs(stack.pop()):
                if q in remaining_all:
                    remaining_all.discard(q)
                    comp.add(q)
                    stack.append(q)
        comps.append(comp)

    paths = []
    for comp in comps:
        remaining = set(comp)
        ends = [p for p in comp if deg.get(p) == 1]
        start = min(ends, key=lambda p: (p[0], p[1])) if ends else min(comp)
        cur = start
        while remaining:
            if start not in remaining:
                cands = [p for p in remaining if deg.get(p) == 1] or list(remaining)
                start = min(cands, key=lambda p: (p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2)
            path, cur, d = [start], start, None
            remaining.discard(start)
            while True:
                nx = [q for q in nbrs(cur) if q in remaining]
                if not nx:
                    break
                if d is not None:      # keep going straight when there is a choice
                    nx.sort(key=lambda q: -((q[0] - cur[0]) * d[0] + (q[1] - cur[1]) * d[1]))
                q = nx[0]
                d = (q[0] - cur[0], q[1] - cur[1])
                remaining.discard(q)
                path.append(q)
                cur = q
            if len(path) > 6 and abs(path[0][0] - path[-1][0]) <= 2 and abs(path[0][1] - path[-1][1]) <= 2:
                path.append(path[0])          # closed loop
            paths.append([(float(x), float(y)) for y, x in path])
            start = cur
    return paths


def _length(pl):
    return sum(math.hypot(pl[i + 1][0] - pl[i][0], pl[i + 1][1] - pl[i][1]) for i in range(len(pl) - 1))


def _simplify(pls, eps):
    out = []
    for pl in pls:
        if len(pl) < 2:
            continue
        if eps > 0 and len(pl) > 2:
            arr = np.array(pl, dtype=np.float32).reshape(-1, 1, 2)
            closed = math.hypot(pl[0][0] - pl[-1][0], pl[0][1] - pl[-1][1]) < 1.5 and len(pl) > 6
            arr = cv2.approxPolyDP(arr, eps, closed)
            pl = [(float(p[0][0]), float(p[0][1])) for p in arr]
            if closed and len(pl) > 2 and pl[0] != pl[-1]:
                pl.append(pl[0])
        if len(pl) >= 2:
            out.append(pl)
    return out


def trace_image(path, mode=None):
    """Image file -> (polylines in pixels, width, height)."""
    gray = _load_gray(path)
    h, w = gray.shape[:2]
    mode = mode or config.TRACE_MODE
    if mode == "edges":
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), config.CANNY_LOW, config.CANNY_HIGH)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        pls = [[(float(p[0][0]), float(p[0][1])) for p in c] for c in contours if len(c) >= 2]
    else:
        pls = _trace_skeleton(_skeleton(_ink_mask(gray)))
    pls = _simplify(pls, config.SIMPLIFY_EPS_PX)
    pls = [pl for pl in pls if _length(pl) >= config.MIN_STROKE_PX]
    pls.sort(key=_length, reverse=True)
    return pls[: config.MAX_STROKES], w, h


# --- paper space ----------------------------------------------------------------------------------

def fit_to_paper(pls_px, w, h, W=None, H=None, margin=None):
    W = W or config.PAPER_W_MM
    H = H or config.PAPER_H_MM
    m = config.MARGIN_MM if margin is None else margin
    aw, ah = W - 2 * m, H - 2 * m
    s = min(aw / float(w), ah / float(h))
    ox = m + (aw - w * s) / 2.0
    oy = m + (ah - h * s) / 2.0
    return [[(ox + x * s, oy + (h - y) * s) for x, y in pl] for pl in pls_px]


def clamp(pls, W=None, H=None):
    W = W or config.PAPER_W_MM
    H = H or config.PAPER_H_MM
    return [[(min(max(x, 0.0), W), min(max(y, 0.0), H)) for x, y in pl] for pl in pls if len(pl) >= 2]


def order_strokes(pls):
    """Greedy nearest-neighbour ordering, reversing strokes when that is closer."""
    rest = [list(pl) for pl in pls if len(pl) >= 2]
    out = []
    cur = (0.0, 0.0)
    while rest:
        best, bi, brev = None, -1, False
        for i, pl in enumerate(rest):
            d0 = math.hypot(pl[0][0] - cur[0], pl[0][1] - cur[1])
            d1 = math.hypot(pl[-1][0] - cur[0], pl[-1][1] - cur[1])
            d, rev = (d0, False) if d0 <= d1 else (d1, True)
            if best is None or d < best:
                best, bi, brev = d, i, rev
        pl = rest.pop(bi)
        if brev:
            pl.reverse()
        out.append(pl)
        cur = pl[-1]
    return out


def stats(pls):
    draw = sum(_length(pl) for pl in pls)
    travel, cur = 0.0, (0.0, 0.0)
    for pl in pls:
        travel += math.hypot(pl[0][0] - cur[0], pl[0][1] - cur[1])
        cur = pl[-1]
    travel += math.hypot(cur[0], cur[1])
    if config.PEN_MODE == "servo":
        pen_s = 0.6
    else:
        pen_s = 2.0 * abs(config.PEN_UP_Z - config.PEN_DOWN_Z) / max(config.PEN_FEED, 1) * 60.0
    seconds = (draw / max(config.DRAW_FEED, 1) + travel / max(config.TRAVEL_FEED, 1)) * 60.0 * 1.15 + len(pls) * pen_s
    return {"strokes": len(pls), "points": sum(len(pl) for pl in pls),
            "draw_mm": round(draw, 1), "travel_mm": round(travel, 1), "seconds": int(seconds)}


def to_svg(pls, W=None, H=None):
    W = W or config.PAPER_W_MM
    H = H or config.PAPER_H_MM
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.1f} {H:.1f}" '
             f'width="100%" preserveAspectRatio="xMidYMid meet">',
             f'<rect x="0" y="0" width="{W:.1f}" height="{H:.1f}" fill="#fffdf8"/>']
    for pl in pls:
        pts = " ".join(f"{x:.2f},{H - y:.2f}" for x, y in pl)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#241b1a" stroke-width="0.7" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def to_png(pls, path, px_per_mm=4, W=None, H=None):
    _require_cv2()
    W = W or config.PAPER_W_MM
    H = H or config.PAPER_H_MM
    canvas = np.full((int(H * px_per_mm), int(W * px_per_mm)), 255, np.uint8)
    for pl in pls:
        arr = np.array([(x * px_per_mm, (H - y) * px_per_mm) for x, y in pl], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr], False, 0, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return str(path)
