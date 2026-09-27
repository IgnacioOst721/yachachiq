"""Pen order: draw the strokes with as little pen-up travel as possible.

    optimize(strokes, start=(0, 0), budget_s=2.0) -> reordered (and reversed) strokes
    continuous(strokes, ...)                       -> one route for a pen that never lifts

optimize = greedy nearest neighbour (either end of every stroke) + 2-opt on the
stroke sequence (reversing a block also reverses each stroke in it) + Or-opt
(moving blocks of 1-3 strokes elsewhere, optionally reversed). Everything is
vectorised with numpy; the improvement loops stop at the time budget.
"""
from __future__ import annotations

import math
import time
from typing import List, Sequence, Tuple

import numpy as np

from .geom import Stroke, travel


def _nn(strokes: List[Stroke], start) -> List[Stroke]:
    n = len(strokes)
    S = np.array([s[0] for s in strokes])
    E = np.array([s[-1] for s in strokes])
    alive = np.ones(n, dtype=bool)
    cur = np.asarray(start, dtype=float)
    out = []
    for _ in range(n):
        ds = np.hypot(*(S - cur).T)
        de = np.hypot(*(E - cur).T)
        ds[~alive] = np.inf
        de[~alive] = np.inf
        i, j = int(np.argmin(ds)), int(np.argmin(de))
        if ds[i] <= de[j]:
            s = strokes[i]
            alive[i] = False
        else:
            s = strokes[j][::-1]
            alive[j] = False
        out.append(s)
        cur = s[-1]
    return out


def _two_opt(A: np.ndarray, B: np.ndarray, order: np.ndarray, rev: np.ndarray, p0: np.ndarray,
             deadline: float) -> bool:
    """In-place 2-opt on (A = drawn start, B = drawn end) arrays. Returns True if improved."""
    n = len(A)
    improved_any = False
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            prev = p0 if i == 0 else B[i - 1]
            j = np.arange(i + 1, n)
            nxtA = np.vstack([A[i + 2:], np.full((1, 2), np.nan)])      # A[j+1]
            old = np.hypot(*(prev - A[i])) + np.nan_to_num(np.hypot(*(B[j] - nxtA).T))
            new = np.hypot(*(prev - B[j]).T) + np.nan_to_num(np.hypot(*(A[i] - nxtA).T))
            delta = new - old
            k = int(np.argmin(delta))
            if delta[k] < -1e-6:
                jj = j[k]
                A[i:jj + 1], B[i:jj + 1] = B[i:jj + 1][::-1].copy(), A[i:jj + 1][::-1].copy()
                order[i:jj + 1] = order[i:jj + 1][::-1].copy()
                rev[i:jj + 1] = ~rev[i:jj + 1][::-1]
                improved = improved_any = True
            if time.time() > deadline:
                break
    return improved_any


def _or_opt(A, B, order, rev, p0, deadline, max_block: int = 3) -> bool:
    n = len(A)
    improved_any = False
    for L in range(1, max_block + 1):
        i = 0
        while i + L <= n and time.time() < deadline:
            prev = p0 if i == 0 else B[i - 1]
            nxt = A[i + L] if i + L < n else None
            a, b = A[i].copy(), B[i + L - 1].copy()
            gain = np.hypot(*(prev - a)) + (np.hypot(*(b - nxt)) if nxt is not None else 0.0) \
                - (np.hypot(*(prev - nxt)) if nxt is not None else 0.0)
            # remaining sequence without the block; insertion between q and q+1 (q = -1: after p0)
            keep = np.r_[0:i, i + L:n]
            if not len(keep):
                break
            RA, RB = A[keep], B[keep]
            left = np.vstack([p0[None], RB])                   # point before each gap
            right = np.vstack([RA, np.full((1, 2), np.nan)])   # point after each gap
            base = np.nan_to_num(np.hypot(*(left - right).T))
            fwd = np.hypot(*(left - a).T) + np.nan_to_num(np.hypot(*(b - right).T)) - base
            bwd = np.hypot(*(left - b).T) + np.nan_to_num(np.hypot(*(a - right).T)) - base
            gap_orig = i                                        # re-inserting where it was = no change
            fwd[gap_orig] = np.inf
            bwd[gap_orig] = np.inf
            kf, kb = int(np.argmin(fwd)), int(np.argmin(bwd))
            best, k, reverse = (fwd[kf], kf, False) if fwd[kf] <= bwd[kb] else (bwd[kb], kb, True)
            if best < gain - 1e-6:
                blkA, blkB = A[i:i + L].copy(), B[i:i + L].copy()
                blkO, blkR = order[i:i + L].copy(), rev[i:i + L].copy()
                if reverse:
                    blkA, blkB = blkB[::-1].copy(), blkA[::-1].copy()
                    blkO, blkR = blkO[::-1].copy(), ~blkR[::-1]
                RO, RR = order[keep], rev[keep]
                A[:] = np.vstack([RA[:k], blkA, RA[k:]])
                B[:] = np.vstack([RB[:k], blkB, RB[k:]])
                order[:] = np.concatenate([RO[:k], blkO, RO[k:]])
                rev[:] = np.concatenate([RR[:k], blkR, RR[k:]])
                improved_any = True
            else:
                i += 1
    return improved_any


def optimize(strokes: Sequence[Stroke], start=(0.0, 0.0), budget_s: float = 2.0) -> List[Stroke]:
    """Reorder/reverse strokes to minimise pen-up travel starting at `start`."""
    strokes = [np.asarray(s, dtype=float) for s in strokes if len(s)]
    if len(strokes) < 2:
        return strokes
    deadline = time.time() + budget_s
    seq = _nn(strokes, start)
    A = np.array([s[0] for s in seq])
    B = np.array([s[-1] for s in seq])
    order = np.arange(len(seq))
    rev = np.zeros(len(seq), dtype=bool)
    p0 = np.asarray(start, dtype=float)
    for _ in range(50):
        a = _two_opt(A, B, order, rev, p0, deadline)
        b = _or_opt(A, B, order, rev, p0, deadline)
        if not (a or b) or time.time() > deadline:
            break
    return join_touching([seq[o][::-1] if r else seq[o] for o, r in zip(order, rev)])


def report(original: Sequence[Stroke], optimized: Sequence[Stroke], start=(0.0, 0.0)) -> dict:
    return {"travel_naive_mm": round(travel(original, start), 1),
            "travel_mm": round(travel(optimized, start), 1)}


def _point_grid(strokes: List[Stroke], cell: float) -> dict:
    grid: dict = {}
    for si, s in enumerate(strokes):
        for k, (x, y) in enumerate(s):
            grid.setdefault((int(x // cell), int(y // cell)), []).append((si, k))
    return grid


def continuous(strokes: Sequence[Stroke], start=(0.0, 0.0), join_tol: float = 0.6,
               max_retrace: float = 400.0) -> Tuple[List[Stroke], float]:
    """Route for a pen that NEVER lifts (ported from v1 plan_continuous, made fast).

    Strokes that touch (an end within join_tol of another stroke) form groups; each group is
    drawn as one walk: draw a stroke, and for every undrawn stroke hanging off it, retrace
    along drawn ink to the junction, draw the child (recursively) and come back along its
    own ink. Retracing adds no new mark. Between groups the pen hops; before a long hop it
    first walks back along drawn ink to the drawn point nearest the next group (up to
    max_retrace mm), so the visible connecting line is as short as possible.
    Returns ([route], visible_hop_mm)."""
    import sys
    pls = [np.asarray(s, dtype=float) for s in strokes if len(s) >= 1]
    if not pls:
        return [], 0.0
    sys.setrecursionlimit(max(10000, 4 * len(pls) + 100))
    grid = _point_grid(pls, max(join_tol, 1e-3))
    touch: List[list] = [[] for _ in pls]             # touch[b] = [(a, end_of_a, k_on_b)]
    for a, s in enumerate(pls):
        for end in (0, len(s) - 1):
            p = s[end]
            gx, gy = int(p[0] // join_tol), int(p[1] // join_tol)
            best: dict = {}
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for (b, k) in grid.get((gx + dx, gy + dy), ()):
                        if b == a:
                            continue
                        d = float(np.hypot(*(pls[b][k] - p)))
                        if d <= join_tol and (b not in best or d < best[b][0]):
                            best[b] = (d, k)
            for b, (d, k) in best.items():
                touch[b].append((a, end, k))
    drawn = [False] * len(pls)
    route: List[np.ndarray] = []

    def emit(pts):
        for p in pts:
            if not route or float(np.hypot(*(route[-1] - p))) > 1e-9:
                route.append(p)

    def walk(i: int, forward: bool):
        pl = pls[i] if forward else pls[i][::-1]
        drawn[i] = True
        emit(pl)
        kids = []
        for (a, end, k) in touch[i]:
            if not drawn[a]:
                kk = k if forward else len(pls[i]) - 1 - k
                kids.append((kk, a, end))
        kids.sort(key=lambda t: -t[0])                # from the far end back toward the start
        pos = len(pl) - 1
        for kk, a, end in kids:
            if drawn[a]:
                continue
            if kk < pos:
                emit(pl[kk:pos][::-1])
            elif kk > pos:
                emit(pl[pos + 1:kk + 1])
            pos = kk
            cpl, cpos = walk(a, forward=(end == 0))
            emit(cpl[:cpos][::-1])                    # back along the child's own ink
            emit([pl[kk]])
        return pl, pos

    ends = np.array([[s[0], s[-1]] for s in pls])     # (n, 2, 2)
    cur = np.asarray(start, dtype=float)
    visible = 0.0
    remaining = np.ones(len(pls), dtype=bool)
    while remaining.any():
        d = np.hypot(ends[:, :, 0] - cur[0], ends[:, :, 1] - cur[1])
        d[~remaining] = np.inf
        flat = int(np.argmin(d))
        i, which = divmod(flat, 2)
        dist = float(d[i, which])
        if route and dist > join_tol:
            target = ends[i, which]
            R = np.asarray(route)
            dd = np.hypot(*(R - target).T)
            bi = int(np.argmin(dd))
            seg = np.hypot(*np.diff(R[bi:], axis=0).T).sum() if bi < len(R) - 1 else 0.0
            if dd[bi] < dist - 1.0 and seg <= max_retrace:
                emit(R[bi:-1][::-1])
                dist = float(dd[bi])
        if route:
            visible += dist
        walk(i, forward=(which == 0))
        cur = route[-1]
        remaining = np.array([not x for x in drawn])
    return [np.asarray(route)], visible


def join_touching(strokes: Sequence[Stroke], tol: float = 0.05) -> List[Stroke]:
    """Merge consecutive strokes when one ends where the next begins (no pen lift needed)."""
    out: List[Stroke] = []
    for s in strokes:
        s = np.asarray(s, dtype=float)
        if not len(s):
            continue
        if out and float(np.hypot(*(out[-1][-1] - s[0]))) <= tol:
            out[-1] = np.vstack([out[-1], s[1:]]) if len(s) > 1 else out[-1]
        else:
            out.append(s)
    return out
