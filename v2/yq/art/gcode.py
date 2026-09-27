"""Pen strokes (paper mm, origin top-left, y down) -> GRBL-style G-code.

    to_gcode(strokes, name) -> list of lines          (raises ValueError when outside the paper)
    estimate(strokes)       -> {"draw_mm","travel_mm","pen_lifts","seconds"}

Pen modes (settings.PEN_MODE): "z" = G1 Z up/down, "servo" = M3 S<angle> + dwell,
"none" = the pen never lifts (use order.continuous first). Machine coordinates:
origin at the paper's bottom-left corner with y up (settings.Y_UP, like v1), optional
axis swap and origin offset. The time estimate simulates GRBL's planner (junction
deviation + trapezoid acceleration), which is much closer to reality than length/feed.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from . import settings
from .geom import Stroke


def _to_machine(x: float, y: float) -> tuple:
    W, H = settings.PAPER_W_MM, settings.PAPER_H_MM
    mx, my = x, (H - y) if settings.Y_UP else y
    if settings.SWAP_XY:
        mx, my = my, mx
    return mx + settings.ORIGIN_X_MM, my + settings.ORIGIN_Y_MM


def origin_page() -> tuple:
    """The machine origin (where the pen starts and parks) in paper coordinates."""
    return (0.0, settings.PAPER_H_MM if settings.Y_UP else 0.0)


def check_bounds(strokes: Sequence[Stroke], tol: float = 1e-6) -> None:
    W, H = settings.PAPER_W_MM, settings.PAPER_H_MM
    for i, s in enumerate(strokes):
        if len(s) and (s[:, 0].min() < -tol or s[:, 1].min() < -tol or s[:, 0].max() > W + tol or s[:, 1].max() > H + tol):
            raise ValueError("stroke %d leaves the %gx%g mm paper: x %.2f..%.2f y %.2f..%.2f"
                             % (i, W, H, s[:, 0].min(), s[:, 0].max(), s[:, 1].min(), s[:, 1].max()))
    mw, mh = settings.MACHINE_W_MM, settings.MACHINE_H_MM
    if mw > 0 and mh > 0:
        for s in strokes:
            for x, y in s:
                mx, my = _to_machine(x, y)
                if not (-tol <= mx <= mw + tol and -tol <= my <= mh + tol):
                    raise ValueError("point (%.2f, %.2f) is outside the machine travel %gx%g mm" % (mx, my, mw, mh))


def _xy(x: float, y: float) -> str:
    mx, my = _to_machine(x, y)
    return "X%.2f Y%.2f" % (mx, my)


def pen_up() -> List[str]:
    if settings.PEN_MODE == "none":
        return []
    if settings.PEN_MODE == "servo":
        return ["M3 S%d" % int(settings.SERVO_UP), "G4 P%.2f" % settings.SERVO_DELAY_S]
    return ["G1 Z%.2f F%d" % (settings.PEN_UP_Z, int(settings.PEN_FEED))]


def pen_down() -> List[str]:
    if settings.PEN_MODE == "none":
        return []
    if settings.PEN_MODE == "servo":
        return ["M3 S%d" % int(settings.SERVO_DOWN), "G4 P%.2f" % settings.SERVO_DELAY_S]
    return ["G1 Z%.2f F%d" % (settings.PEN_DOWN_Z, int(settings.PEN_FEED)), "G4 P0.05"]


def to_gcode(strokes: Sequence[Stroke], name: str = "yachachiq") -> List[str]:
    strokes = [np.asarray(s, dtype=float) for s in strokes if len(s)]
    check_bounds(strokes)
    est = estimate(strokes)
    lines = ["(Yachachiq v2 - %s - %d strokes - %.0f mm ink - ~%.1f min)"
             % (name.replace("(", "[").replace(")", "]"), len(strokes), est["draw_mm"], est["seconds"] / 60.0),
             "(paper %gx%g mm, pen %s, origin bottom-left)" % (settings.PAPER_W_MM, settings.PAPER_H_MM,
                                                               settings.PEN_MODE),
             "G21", "G90", "G94"]
    lines += pen_up()
    feed = int(settings.DRAW_FEED)
    for s in strokes:
        lines.append("G0 " + _xy(*s[0]))
        lines += pen_down()
        if len(s) == 1:
            lines.append("G1 %s F%d" % (_xy(s[0][0] + 0.05, s[0][1]), feed))   # a dot
        for x, y in s[1:]:
            lines.append("G1 %s F%d" % (_xy(x, y), feed))
        lines += pen_up()
    lines.append("G0 " + _xy(*origin_page()))           # park at the origin corner
    lines.append("M5" if settings.PEN_MODE == "servo" else "(end)")
    return lines


def save(lines: List[str], path) -> str:
    with open(path, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def _polyline_time(pts: np.ndarray, vmax: float, a: float, dev: float = 0.01) -> float:
    """GRBL-like planner: junction speeds from the junction deviation, then trapezoids."""
    d = np.diff(pts, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > 1e-6
    d, L = d[keep], L[keep]
    n = len(L)
    if n == 0:
        return 0.0
    u = d / L[:, None]
    vj2 = np.zeros(n + 1)
    for i in range(1, n):
        c = -float(np.dot(u[i - 1], u[i]))
        if c > 0.999999:
            vj2[i] = 0.0
        elif c < -0.999999:
            vj2[i] = vmax * vmax
        else:
            s = math.sqrt(0.5 * (1.0 - c))
            vj2[i] = min(vmax * vmax, a * dev * s / max(1e-9, 1.0 - s))
    for i in range(n - 1, -1, -1):                       # backward: must be able to stop
        vj2[i] = min(vj2[i], vj2[i + 1] + 2 * a * L[i])
    for i in range(n):                                   # forward: can only accelerate so much
        vj2[i + 1] = min(vj2[i + 1], vj2[i] + 2 * a * L[i])
    t = 0.0
    vm2 = vmax * vmax
    for i in range(n):
        v0, v1 = vj2[i], vj2[i + 1]
        vp2 = (2 * a * L[i] + v0 + v1) / 2.0
        if vp2 <= vm2:
            vp = math.sqrt(vp2)
            t += (2 * vp - math.sqrt(v0) - math.sqrt(v1)) / a
        else:
            vm = vmax
            da = (vm2 - v0) / (2 * a)
            dd = (vm2 - v1) / (2 * a)
            t += (vm - math.sqrt(v0)) / a + (vm - math.sqrt(v1)) / a + max(0.0, L[i] - da - dd) / vm
    return t


def estimate(strokes: Sequence[Stroke], start=None) -> dict:
    a = max(1.0, settings.ACCEL_MM_S2)
    draw_v = settings.DRAW_FEED / 60.0
    trav_v = settings.TRAVEL_FEED / 60.0
    if settings.PEN_MODE == "servo":
        lift_s = settings.SERVO_DELAY_S + 0.02
    elif settings.PEN_MODE == "z":
        lift_s = abs(settings.PEN_UP_Z - settings.PEN_DOWN_Z) / max(1.0, settings.PEN_FEED / 60.0) + 0.05
    else:
        lift_s = 0.0
    cur = np.array(origin_page() if start is None else start, dtype=float)
    t = draw = trav = 0.0
    lifts = 0
    for s in strokes:
        s = np.asarray(s, dtype=float)
        if not len(s):
            continue
        hop = float(np.hypot(*(s[0] - cur)))
        trav += hop
        t += _polyline_time(np.array([cur, s[0]]), trav_v, a)
        if settings.PEN_MODE != "none":
            t += 2 * lift_s
            lifts += 1
        draw += float(np.sum(np.hypot(*np.diff(s, axis=0).T))) if len(s) > 1 else 0.0
        t += _polyline_time(s, draw_v, a)
        cur = s[-1]
    return {"draw_mm": round(draw, 1), "travel_mm": round(trav, 1), "pen_lifts": lifts,
            "seconds": round(float(t), 1)}
