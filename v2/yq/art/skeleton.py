"""Skeleton -> graph -> long strokes.

    thin(mask)                               -> 1-px skeleton (uint8 0/1)
    build_graph(skel)                        -> Graph (nodes, edges)
    prune_spurs(graph, halfwidth)            -> removes thinning artefacts
    merge_through_junctions(graph, ...)      -> list of pixel polylines

A node is a connected cluster of skeleton pixels whose neighbour count is not 2
(endpoints have 1, junctions >= 3). Edges are the chains of degree-2 pixels
between nodes. Degree-2 pixels have exactly two neighbours, so walking them is
unambiguous; staircase artefacts become small clusters with two edges, which
the merge step joins back into one stroke.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


def thin(mask: np.ndarray) -> np.ndarray:
    """Guo-Hall thinning (OpenCV contrib when present, the same algorithm in numpy otherwise).
    Guo-Hall, not Zhang-Suen: Zhang-Suen erases 45-degree lines whose edges are clean pixel
    staircases (a whole letter-K arm disappeared in our tests); Guo-Hall keeps them."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    if hasattr(cv2, "ximgproc"):
        sk = (cv2.ximgproc.thinning(np.ascontiguousarray(m * 255),
                                    thinningType=cv2.ximgproc.THINNING_GUOHALL) > 0).astype(np.uint8)
    else:
        sk = _guo_hall(m)
    return minimal_skeleton(sk)


def _guo_hall(m: np.ndarray) -> np.ndarray:
    """Guo & Hall (1989) parallel thinning, the variant used by OpenCV ximgproc."""
    img = np.pad(m.astype(np.uint8), 1)
    while True:
        changed = False
        for it in (0, 1):
            P2 = img[:-2, 1:-1]; P3 = img[:-2, 2:]; P4 = img[1:-1, 2:]; P5 = img[2:, 2:]
            P6 = img[2:, 1:-1]; P7 = img[2:, :-2]; P8 = img[1:-1, :-2]; P9 = img[:-2, :-2]
            c = img[1:-1, 1:-1]
            C = ((1 - P2) & (P3 | P4)).astype(np.int16) + ((1 - P4) & (P5 | P6)) + ((1 - P6) & (P7 | P8)) \
                + ((1 - P8) & (P9 | P2))
            N1 = (P9 | P2).astype(np.int16) + (P3 | P4) + (P5 | P6) + (P7 | P8)
            N2 = (P2 | P3).astype(np.int16) + (P4 | P5) + (P6 | P7) + (P8 | P9)
            N = np.minimum(N1, N2)
            if it == 0:
                mm = (P6 | P7 | (1 - P9)) & P8
            else:
                mm = (P2 | P3 | (1 - P5)) & P4
            rm = (c == 1) & (C == 1) & (N >= 2) & (N <= 3) & (mm == 0)
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1]


# 3x3 neighbour positions (dy, dx), clockwise from the top-left corner
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _components(cells, adjacent) -> list:
    cells = list(cells)
    comps, seen = [], set()
    for c in cells:
        if c in seen:
            continue
        comp, stack = [], [c]
        seen.add(c)
        while stack:
            a = stack.pop()
            comp.append(a)
            for b in cells:
                if b not in seen and adjacent(a, b):
                    seen.add(b)
                    stack.append(b)
        comps.append(comp)
    return comps


def _simple_lut() -> np.ndarray:
    """LUT[code] = True when the centre pixel is a staircase corner that can be removed without
    changing the topology (8-connected foreground, 4-connected background)."""
    lut = np.zeros(256, dtype=bool)
    adj8 = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
    adj4 = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    for code in range(256):
        fg = [_NB[i] for i in range(8) if code >> i & 1]
        bg = [_NB[i] for i in range(8) if not code >> i & 1]
        if len(fg) < 2:
            continue
        # only corner pixels of a 4-connected staircase (a vertical AND a horizontal 4-neighbour):
        # removing plain simple points would also eat free line ends pixel by pixel
        vert = bool(code >> 1 & 1 or code >> 5 & 1)
        horiz = bool(code >> 3 & 1 or code >> 7 & 1)
        if not (vert and horiz):
            continue
        c8 = len(_components(fg, adj8))
        c4 = [c for c in _components(bg, adj4) if any(abs(y) + abs(x) == 1 for y, x in c)]
        lut[code] = c8 == 1 and len(c4) == 1
    return lut


_LUT = None


def minimal_skeleton(sk: np.ndarray) -> np.ndarray:
    """Remove redundant pixels (staircase corners, 2-px-thick diagonal runs) so every
    remaining line pixel has exactly two neighbours. Four interleaved sub-fields keep the
    parallel removal safe (no two pixels of one sub-field touch)."""
    global _LUT
    if _LUT is None:
        _LUT = _simple_lut()
    img = np.pad((sk > 0).astype(np.uint8), 1)
    H, W = img.shape
    yy, xx = np.mgrid[0:H - 2, 0:W - 2]
    fields = [((yy % 2) == a) & ((xx % 2) == b) for a in (0, 1) for b in (0, 1)]
    for _ in range(20):
        changed = False
        for f in fields:
            c = img[1:-1, 1:-1]
            code = np.zeros(c.shape, dtype=np.int32)
            for i, (dy, dx) in enumerate(_NB):
                code |= img[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx].astype(np.int32) << i
            rm = (c > 0) & f & _LUT[code]
            if rm.any():
                c[rm] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1]


def _zhang_suen(m: np.ndarray) -> np.ndarray:
    img = np.pad(m.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P2 = img[:-2, 1:-1]; P3 = img[:-2, 2:]; P4 = img[1:-1, 2:]; P5 = img[2:, 2:]
            P6 = img[2:, 1:-1]; P7 = img[2:, :-2]; P8 = img[1:-1, :-2]; P9 = img[:-2, :-2]
            c = img[1:-1, 1:-1]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(x.astype(np.int16) for x in nb)
            seq = nb + [P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1]


@dataclass
class Edge:
    n0: int
    n1: int
    pts: np.ndarray                 # (N, 2) float, x/y pixels, from n0 to n1
    alive: bool = True

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(*np.diff(self.pts, axis=0).T))) if len(self.pts) > 1 else 0.0


@dataclass
class Graph:
    nodes: np.ndarray               # (K, 2) node positions (x, y)
    node_px: np.ndarray             # (K,) pixel count of each node cluster
    edges: List[Edge] = field(default_factory=list)

    def degree(self) -> np.ndarray:
        d = np.zeros(len(self.nodes), dtype=int)
        for e in self.edges:
            if e.alive:
                d[e.n0] += 1
                d[e.n1] += 1
        return d


def build_graph(skel: np.ndarray) -> Graph:
    import cv2
    sk = np.pad((skel > 0).astype(np.uint8), 1)
    H, W = sk.shape
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    deg = cv2.filter2D(sk.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT).round().astype(np.int32) * sk
    node_mask = ((sk > 0) & (deg != 2)).astype(np.uint8)
    num, lab = cv2.connectedComponents(node_mask, connectivity=8)
    flat_sk = sk.ravel()
    flat_lab = lab.ravel()
    offs = np.array([-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1])
    # node positions = centroid of each cluster (minus the 1-px padding)
    ys, xs = np.nonzero(node_mask)
    ids = lab[ys, xs]
    cnt = np.bincount(ids, minlength=num).astype(float)
    cx = np.bincount(ids, weights=xs, minlength=num) / np.maximum(cnt, 1)
    cy = np.bincount(ids, weights=ys, minlength=num) / np.maximum(cnt, 1)
    # node 0 of the label image is "no node"; graph node i = label i+1
    nodes = np.stack([cx[1:] - 1, cy[1:] - 1], axis=1) if num > 1 else np.zeros((0, 2))
    g = Graph(nodes=nodes, node_px=cnt[1:].astype(int) if num > 1 else np.zeros(0, int))
    visited = np.zeros(H * W, dtype=bool)

    def xy(p):
        return (p % W - 1, p // W - 1)

    def walk(start_node_px: int, first: int) -> None:
        path = [start_node_px, first]
        visited[first] = True
        prev, cur = start_node_px, first
        while True:
            nxt = -1
            for o in offs:
                q = cur + o
                if flat_sk[q] and q != prev:
                    nxt = q
                    break
            if nxt < 0:                                 # dead end inside an edge (should not happen)
                end = -1
                break
            if flat_lab[nxt]:
                path.append(nxt)
                end = flat_lab[nxt] - 1
                break
            if visited[nxt]:
                end = -1
                break
            visited[nxt] = True
            path.append(nxt)
            prev, cur = cur, nxt
        n0 = flat_lab[start_node_px] - 1
        pts = np.array([xy(p) for p in path], dtype=float)
        if end < 0:                                     # broken walk: finish at the last pixel as a new node
            g.nodes = np.vstack([g.nodes, pts[-1:]])
            g.node_px = np.append(g.node_px, 1)
            end = len(g.nodes) - 1
        if end == n0 and len(pts) <= 4:                 # 2-3 px loop hugging a junction cluster
            return
        _snap_ends(g, pts, n0, end)
        g.edges.append(Edge(n0, end, pts))

    node_pixels = np.flatnonzero(node_mask.ravel())
    for p in node_pixels:
        for o in offs:
            q = p + o
            if flat_sk[q] and not flat_lab[q] and not visited[q]:
                walk(p, q)
    # isolated cycles made only of degree-2 pixels
    rest = np.flatnonzero((flat_sk > 0) & (flat_lab == 0) & ~visited)
    for p in rest:
        if visited[p]:
            continue
        path = [p]
        visited[p] = True
        prev, cur = -1, p
        while True:
            nxt = -1
            for o in offs:
                q = cur + o
                if flat_sk[q] and q != prev and not visited[q]:
                    nxt = q
                    break
            if nxt < 0:
                break
            visited[nxt] = True
            path.append(nxt)
            prev, cur = cur, nxt
        path.append(p)
        pts = np.array([xy(q) for q in path], dtype=float)
        g.nodes = np.vstack([g.nodes, pts[:1]])
        g.node_px = np.append(g.node_px, 1)
        nid = len(g.nodes) - 1
        g.edges.append(Edge(nid, nid, pts))
    return g


def _snap_ends(g: Graph, pts: np.ndarray, n0: int, n1: int) -> None:
    """Junction clusters: start/end the edge at the cluster centroid so strokes meet cleanly."""
    if g.node_px[n0] > 1:
        pts[0] = g.nodes[n0]
    if g.node_px[n1] > 1:
        pts[-1] = g.nodes[n1]


def prune_spurs(g: Graph, halfwidth: np.ndarray, factor: float = 1.8, extra_px: float = 2.0,
                rounds: int = 3) -> int:
    """Remove short dead-end branches that hang off a junction (thinning artefacts).

    halfwidth: distance-transform image (ink half-width at each pixel). A branch is a
    spur when it is shorter than factor * local half-width + extra_px, so thick lines
    lose their end/corner whiskers while real small details on thin lines survive.
    Shortest spurs go first so only one arm of a short fork is removed."""
    removed = 0
    h, w = halfwidth.shape
    for _ in range(rounds):
        deg = g.degree()
        cands = []
        for e in g.edges:
            if not e.alive or e.n0 == e.n1:
                continue
            for tip, base in ((e.n0, e.n1), (e.n1, e.n0)):
                if deg[tip] == 1 and deg[base] >= 3:
                    # the width of the branch ITSELF (not of the junction blob, which is wide where
                    # lines meet at an acute angle): real short arms keep their length
                    xs = np.clip(np.round(e.pts[:, 0]).astype(int), 0, w - 1)
                    ys = np.clip(np.round(e.pts[:, 1]).astype(int), 0, h - 1)
                    hw = float(np.median(halfwidth[ys, xs]))
                    if e.length < factor * hw + extra_px and not _is_stub(g, e, base, hw):
                        cands.append((e.length, id(e), e, base))
                    break
        if not cands:
            break
        cands.sort(key=lambda t: t[0])
        changed = False
        for _, _, e, base in cands:
            deg = g.degree()
            if deg[base] >= 3 and e.alive:
                e.alive = False
                removed += 1
                changed = True
        if not changed:
            break
    return removed


def _is_stub(g: Graph, spur: Edge, base: int, hw: float) -> bool:
    """A short branch that straightly continues exactly one other branch (the top of the stem of
    an 'a' above its bowl) is real; a thinning spur points into a corner, between the branches.
    Test: the branch tip lies on the extension of exactly one other branch's straight part."""
    sp = spur.pts if spur.n0 == base else spur.pts[::-1]
    tip = sp[-1]
    acrosses = []
    for e in g.edges:
        if e is spur or not e.alive or base not in (e.n0, e.n1) or e.n0 == e.n1:
            continue
        p = e.pts if e.n0 == base else e.pts[::-1]
        d = np.hypot(*(p - p[0]).T)
        part = p[(d >= 2.0 * hw) & (d <= 8.0 * hw)]
        if len(part) < 3:
            part = p[d >= hw]
        if len(part) < 2:
            continue
        c = part.mean(axis=0)
        _, _, vt = np.linalg.svd(part - c, full_matrices=False)
        u = vt[0]
        if float(np.dot(part[-1] - part[0], u)) > 0:
            u = -u                                    # u points from the branch towards the base
        off = tip - c
        along = float(np.dot(off, u))
        across = abs(float(off[0] * u[1] - off[1] * u[0]))
        if along > 0:
            acrosses.append(across)
    acrosses.sort()
    if not acrosses or acrosses[0] > max(1.0, 0.6 * hw):
        return False
    # it must continue ONE branch clearly better than any other (a corner spur sits
    # symmetrically between two branches)
    return len(acrosses) == 1 or acrosses[1] >= max(2.0 * acrosses[0], acrosses[0] + 1.5)


def _edge_dir(pts: np.ndarray, hw: float) -> np.ndarray:
    """Direction leaving pts[0], ignoring the first ~2 half-widths where a junction bends the
    skeleton: straight-line fit of the part 2..8 half-widths away (whole edge if shorter)."""
    d = np.hypot(*(pts - pts[0]).T)
    part = pts[(d >= 2.0 * hw) & (d <= 8.0 * hw)]
    if len(part) < 3:
        v = pts[-1] - pts[0]
        n = math.hypot(v[0], v[1])
        return v / n if n > 1e-9 else np.zeros(2)
    c = part.mean(axis=0)
    _, _, vt = np.linalg.svd(part - c, full_matrices=False)
    u = vt[0]
    if float(np.dot(c - pts[0], u)) < 0:
        u = -u
    return u


def _direction(pts: np.ndarray, reach: float) -> np.ndarray:
    """Unit direction leaving pts[0], measured `reach` pixels along the polyline."""
    d = np.hypot(*(pts - pts[0]).T)
    idx = np.flatnonzero(d >= reach)
    far = pts[idx[0]] if len(idx) else pts[-1]
    v = far - pts[0]
    n = math.hypot(v[0], v[1])
    return v / n if n > 1e-9 else np.zeros(2)


def merge_through_junctions(g: Graph, reach: float = 8.0, max_turn_deg: float = 50.0,
                            hw: float = None) -> List[np.ndarray]:
    """Chain edges into long strokes: at every node, pair the edge ends that continue most
    straight (turn <= max_turn_deg); degree-2 nodes always join. Returns pixel polylines."""
    edges = [e for e in g.edges if e.alive and len(e.pts) >= 2]
    ends_at: Dict[int, List[Tuple[int, int]]] = {}
    for i, e in enumerate(edges):
        ends_at.setdefault(e.n0, []).append((i, 0))
        ends_at.setdefault(e.n1, []).append((i, 1))
    pair: Dict[Tuple[int, int], Tuple[int, int]] = {}
    min_straight = math.cos(math.radians(max_turn_deg))
    for node, ends in ends_at.items():
        if len(ends) < 2:
            continue
        dirs = []
        for (i, which) in ends:
            p = edges[i].pts if which == 0 else edges[i].pts[::-1]
            dirs.append(_edge_dir(p, hw) if hw else _direction(p, reach))
        if len(ends) == 2 and ends[0][0] != ends[1][0]:
            pair[ends[0]] = ends[1]
            pair[ends[1]] = ends[0]
            continue
        cands = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                if ends[a][0] == ends[b][0]:
                    continue                            # a loop edge: its two ends stay open here
                straight = -float(np.dot(dirs[a], dirs[b]))
                if straight >= min_straight:
                    cands.append((straight, a, b))
        cands.sort(reverse=True)
        used = set()
        for straight, a, b in cands:
            if a in used or b in used:
                continue
            used.update((a, b))
            pair[ends[a]] = ends[b]
            pair[ends[b]] = ends[a]
    done = [False] * len(edges)
    strokes: List[np.ndarray] = []

    def follow(i: int, enter: int) -> List[np.ndarray]:
        parts = []
        start = (i, enter)
        while True:
            done[i] = True
            p = edges[i].pts if enter == 0 else edges[i].pts[::-1]
            parts.append(p if not parts else p[1:])
            nxt = pair.get((i, 1 - enter))
            if nxt is None or done[nxt[0]]:
                if nxt is not None and nxt == start:
                    parts.append(parts[0][:1])          # closed loop: end where it started
                break
            i, enter = nxt
        return parts

    # open chains first: start at an edge end that is not paired
    for i, e in enumerate(edges):
        for which in (0, 1):
            if not done[i] and (i, which) not in pair:
                strokes.append(np.concatenate(follow(i, which), axis=0))
    for i in range(len(edges)):                         # what is left are closed cycles
        if not done[i]:
            strokes.append(np.concatenate(follow(i, 0), axis=0))
    return strokes


def contract_short_edges(g: Graph, halfwidth: np.ndarray, factor: float = 2.2, extra_px: float = 2.0) -> int:
    """Two thick lines crossing thin into two Y junctions joined by a short bridge.
    Contract such bridges (both ends junctions, short relative to the local width) so the
    crossing becomes one node with four arms that the merge step can pair straight."""
    h, w = halfwidth.shape
    parent = list(range(len(g.nodes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    deg = g.degree()
    n = 0
    for e in sorted((e for e in g.edges if e.alive and e.n0 != e.n1), key=lambda e: e.length):
        if deg[e.n0] < 3 or deg[e.n1] < 3:
            continue
        mid = e.pts[len(e.pts) // 2]
        hw = float(halfwidth[min(h - 1, max(0, int(round(mid[1])))), min(w - 1, max(0, int(round(mid[0]))))])
        if e.length < factor * hw + extra_px:
            a, b = find(e.n0), find(e.n1)
            if a != b:
                parent[b] = a
                e.alive = False
                n += 1
    if not n:
        return 0
    groups: Dict[int, list] = {}
    for i in range(len(g.nodes)):
        groups.setdefault(find(i), []).append(i)
    for root, members in groups.items():
        if len(members) > 1:
            pos = g.nodes[members].mean(axis=0)
            g.nodes[root] = pos
            g.node_px[root] = max(2, int(g.node_px[members].sum()))
    for e in g.edges:
        if not e.alive:
            continue
        r0, r1 = find(e.n0), find(e.n1)
        if r0 != e.n0 or len(groups[r0]) > 1:
            e.pts[0] = g.nodes[r0]
        if r1 != e.n1 or len(groups[r1]) > 1:
            e.pts[-1] = g.nodes[r1]
        e.n0, e.n1 = r0, r1
    return n


def free_ends(g: Graph) -> set:
    """Positions (rounded) of endpoint nodes (degree 1): the only ends that may be extended."""
    deg = g.degree()
    return {(round(float(x), 3), round(float(y), 3)) for (x, y), d in zip(g.nodes, deg) if d == 1}


def extend_free_ends(strokes: List[np.ndarray], ends: set, mask: np.ndarray, dist: np.ndarray,
                     reach: float = 6.0) -> List[np.ndarray]:
    """Thinning stops about one half-width short of a round line end: walk each free end
    outwards to the ink boundary and stop one local radius before it."""
    h, w = mask.shape
    out = []
    for s in strokes:
        s = s.copy()
        for head in (True, False):
            if len(s) < 2:
                break
            p = s[0] if head else s[-1]
            if (round(float(p[0]), 3), round(float(p[1]), 3)) not in ends:
                continue
            q = s if head else s[::-1]
            d = _direction(q, reach)
            d = -d                                      # pointing out of the stroke
            if not d.any():
                continue
            back = q[: max(2, min(len(q), int(reach)))]
            r = max(0.0, max(float(dist[min(h - 1, int(round(y))), min(w - 1, int(round(x)))]) for x, y in back) - 0.5)
            t = 0.0
            while t < 4 * r + 4:
                x, y = p + (t + 0.5) * d
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < w and 0 <= yi < h) or not mask[yi, xi]:
                    break
                t += 0.5
            ext = t - r
            if ext > 0.3:
                new = p + ext * d
                s = np.vstack([new[None], s]) if head else np.vstack([s, new[None]])
        out.append(s)
    return out
