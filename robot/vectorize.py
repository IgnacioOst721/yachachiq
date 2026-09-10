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


def _fills_to_outlines(mask, thick_px=None):
    """Split the ink into thin strokes (to skeletonize) and thick/filled regions (to outline).

    Skeletonizing a filled area produces a thicket of little branches - dozens of strokes for
    one black shape. A pen plotter should draw the shape's CONTOUR instead: one line around it.
    Returns (thin_mask, outline_polylines).
    """
    thick_px = thick_px or config.FILL_THICK_PX
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    core = (dist > thick_px).astype(np.uint8) * 255          # pixels deep inside a fat region
    if cv2.countNonZero(core) == 0:
        return mask, []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thick_px + 1, 2 * thick_px + 1))
    thick = cv2.dilate(core, k)                                # grow the core back to the region
    thick = cv2.bitwise_and(thick, mask)
    contours, _ = cv2.findContours(thick, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outlines = []
    for c in contours:
        if len(c) >= 8:
            pts = [(float(p[0][0]), float(p[0][1])) for p in c]
            pts.append(pts[0])                                 # close the loop
            outlines.append(pts)
    thin = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.dilate(thick, np.ones((3, 3), np.uint8))))
    return thin, outlines


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


def _drop_frame(pls, w, h, cover=None):
    """Remove strokes whose bounding box spans most of the image: that is a frame/border."""
    cover = cover or config.FRAME_MIN_COVER
    out = []
    for pl in pls:
        xs = [p[0] for p in pl]; ys = [p[1] for p in pl]
        bw, bh = (max(xs) - min(xs)) / float(w), (max(ys) - min(ys)) / float(h)
        if bw >= cover and bh >= cover:
            continue
        # also the straight edge-hugging pieces of a broken frame
        near_edge = all(min(x, w - x) < 0.06 * w or min(y, h - y) < 0.06 * h for x, y in pl)
        if near_edge and (bw >= 0.5 or bh >= 0.5):
            continue
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
        mask = _ink_mask(gray)
        pls = _trace_skeleton(_skeleton(mask))
        if config.FILL_TO_OUTLINE:
            # try the fill->contour version too and keep whichever the pen draws faster
            thin, outlines = _fills_to_outlines(mask)
            alt = _trace_skeleton(_skeleton(thin)) + outlines
            if sum(_length(pl) for pl in alt) < sum(_length(pl) for pl in pls):
                pls = alt
    if config.DROP_FRAME:
        pls = _drop_frame(pls, w, h)
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


# set by the pipeline once the board is connected: {'max_feed': mm/min, 'accel': mm/s^2}
MACHINE_LIMITS = {}


def _segment_seconds(d, feed_mm_min, vmax_mm_min, accel):
    """How long GRBL really takes for one move: it accelerates and decelerates on every
    segment, and short segments never reach the requested speed at all."""
    if d <= 0:
        return 0.0
    v = min(feed_mm_min, vmax_mm_min) / 60.0            # mm/s actually reachable
    d_acc = v * v / (2.0 * accel)                       # distance needed to reach it
    if 2.0 * d_acc >= d:                                # triangular profile
        return 2.0 * math.sqrt(d / accel)
    return 2.0 * (v / accel) + (d - 2.0 * d_acc) / v



# --- pen that never lifts: one continuous route ------------------------------------------------

def _nearest_on(pl, pt):
    """(index, distance) of the point of polyline pl closest to pt."""
    best = (0, float("inf"))
    for k, q in enumerate(pl):
        d = math.hypot(q[0] - pt[0], q[1] - pt[1])
        if d < best[1]:
            best = (k, d)
    return best


def plan_continuous(pls, jump_ok=None, tol=None):
    """Route for a pen that NEVER lifts.

    Skeleton strokes meet at junctions (a branch usually ends on the middle of another stroke).
    Each connected group of strokes is drawn as one walk: draw a stroke, and for every undrawn
    stroke touching it, retrace along the already-drawn ink to the junction, draw the child
    (recursively), and retrace back. Retracing leaves no new mark. Only the hops between
    groups that do not touch are visible, and those are ordered nearest-first.

    Returns ([route_polyline], visible_hop_mm)."""
    jump_ok = config.JUMP_OK_MM if jump_ok is None else jump_ok
    tol = config.JOIN_TOL_MM if tol is None else tol
    pls = [list(pl) for pl in pls if len(pl) >= 2]
    n = len(pls)
    if not n:
        return [], 0.0
    # junctions: endpoint of stroke a touches point k of stroke b
    touch = [[] for _ in range(n)]                     # touch[b] = [(a, end_of_a, k_on_b)]
    for a in range(n):
        for end in (0, len(pls[a]) - 1):
            pt = pls[a][end]
            for b in range(n):
                if a == b:
                    continue
                k, d = _nearest_on(pls[b], pt)
                if d <= tol:
                    touch[b].append((a, end, k))
    drawn = [False] * n
    route = []
    visible = 0.0

    def walk(i, forward):
        """Draw stroke i (from its start if forward), visiting touching children on the way back."""
        pl = pls[i] if forward else pls[i][::-1]
        drawn[i] = True
        route.extend(pl if not route or route[-1] != pl[0] else pl[1:])
        # children hanging off this stroke, in the order we pass them walking back to the start
        kids = []
        for (a, end, k) in touch[i]:
            if not drawn[a]:
                kk = k if forward else len(pls[i]) - 1 - k
                kids.append((kk, a, end))
        kids.sort(key=lambda t: -t[0])                # from the far end back towards the start
        pos = len(pl) - 1
        for kk, a, end in kids:
            if drawn[a]:
                continue
            # retrace along this stroke from pos back to the junction kk (no new ink)
            if kk < pos:
                route.extend(pl[kk:pos][::-1])
            elif kk > pos:
                route.extend(pl[pos + 1:kk + 1])
            pos = kk
            cpl, cpos = walk(a, forward=(end == 0))   # child starts at the end that touches us
            # come back along the child's own ink to the junction (its grandchildren already
            # returned to the child's line the same way), never with a straight line
            route.extend(cpl[:cpos][::-1])
            route.append(pl[kk])
        return pl, pos                                # where on this stroke the pen ended

    remaining = list(range(n))
    cur = (0.0, 0.0)
    while remaining:
        # nearest undrawn stroke end to where the pen is
        best = None
        for i in remaining:
            for fwd, pt in ((True, pls[i][0]), (False, pls[i][-1])):
                d = math.hypot(pt[0] - cur[0], pt[1] - cur[1])
                if best is None or d < best[0]:
                    best = (d, i, fwd)
        d, i, fwd = best
        if route and d > jump_ok:
            # instead of a straight line across the picture, go BACK over the ink already drawn
            # to the drawn point nearest the new group, and hop from there (shortest visible gap)
            target = pls[i][0] if fwd else pls[i][-1]
            step = max(1, len(route) // 4000)
            bi, bd = len(route) - 1, d
            for k in range(0, len(route), step):
                q = route[k]
                dk = math.hypot(q[0] - target[0], q[1] - target[1])
                if dk < bd:
                    bi, bd = k, dk
            retrace = sum(math.hypot(route[k + 1][0] - route[k][0], route[k + 1][1] - route[k][1])
                          for k in range(bi, len(route) - 1))
            if bd < d - 2.0 and retrace <= config.MAX_RETRACE_MM:
                route.extend(route[bi:len(route) - 1][::-1])   # walk back along the drawn route
                d = bd
        if route:
            visible += d                               # what is left is a real gap in the picture
        walk(i, fwd)
        cur = route[-1]
        remaining = [k for k in remaining if not drawn[k]]
    return [route], visible


def stats(pls, machine=None):
    """Stroke counts and an HONEST time estimate.

    `machine` is what the board actually allows: {"max_feed": mm/min, "accel": mm/s^2}, read
    from GRBL at connect time. Without it we fall back to the configured feeds, which is what
    the estimate used to do - and it lied by half, because the team's machine is set to
    10 mm/s^2 (GRBL's minimum) and spends most of a drawing accelerating.
    """
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

    machine = machine if machine is not None else MACHINE_LIMITS
    vmax = (machine or {}).get("max_feed") or 0
    accel = (machine or {}).get("accel") or 0
    if vmax and accel:
        seconds = 0.0
        cur = (0.0, 0.0)
        for pl in pls:
            seconds += _segment_seconds(math.hypot(pl[0][0] - cur[0], pl[0][1] - cur[1]),
                                        config.TRAVEL_FEED, vmax, accel) + pen_s
            for a, b in zip(pl, pl[1:]):
                seconds += _segment_seconds(math.hypot(b[0] - a[0], b[1] - a[1]),
                                            config.DRAW_FEED, vmax, accel)
            cur = pl[-1]
        seconds += _segment_seconds(math.hypot(cur[0], cur[1]), config.TRAVEL_FEED, vmax, accel)
    else:
        seconds = (draw / max(config.DRAW_FEED, 1) + travel / max(config.TRAVEL_FEED, 1)) * 60.0 * 1.15 \
                  + len(pls) * pen_s
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
