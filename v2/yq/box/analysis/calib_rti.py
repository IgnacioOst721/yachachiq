"""RTI light positions from a chrome (mirror) sphere.

Put a chrome ball of known diameter on the platter (off-centre) and take one photo per LED with
camera B (plus one with the COB light for the outline), at several platter angles. For every photo:
  1. find the ball outline (Hough circle) -> its 3D centre from the known radius and the camera model,
  2. find the specular highlight (brightest small blob inside the ball),
  3. reflect the camera ray on the ball: the LED lies on that reflected ray.
Rays from several ball positions are intersected by least squares -> 3D LED position (box frame,
= object frame at platter 0). With a single position we fall back to the CAD LED height.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import Camera, nominal_leds, unit


def find_ball(outline: np.ndarray, radius_mm: float, cam: Camera) -> Optional[tuple]:
    """-> (centre_3d (3,) box frame, (u, v, r_px)) from a photo where the ball is visible (COB light)."""
    import cv2
    g = outline if outline.ndim == 2 else cv2.cvtColor(outline, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 2)
    r_exp = cam.focal * radius_mm / 300.0
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=g.shape[0], param1=80, param2=25,
                               minRadius=int(r_exp * 0.5), maxRadius=int(r_exp * 2.0))
    if circles is None:
        return None
    u, v, r = circles[0][0]
    o, d = cam.rays(np.array([[u, v]], float))
    # distance to the centre so that the ball of radius R subtends r pixels: sin(a) = R / D, tan(a) ~ r / f
    a = np.arctan(r / cam.focal)
    D = radius_mm / np.sin(a)
    return o + D * d[0], (float(u), float(v), float(r))


def find_highlight(img: np.ndarray, uvr: tuple, min_value: int = 180) -> Optional[tuple]:
    import cv2
    g = img if img.ndim == 2 else img.max(axis=2)
    u, v, r = uvr
    m = np.zeros(g.shape, np.uint8)
    cv2.circle(m, (int(round(u)), int(round(v))), int(r * 0.97), 1, -1)
    gg = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0) * m
    peak = float(gg.max())
    if peak < min_value:
        return None
    sel = gg >= 0.8 * peak
    ys, xs = np.nonzero(sel)
    w = gg[sel]
    return float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())


def reflected_ray(cam: Camera, uv: tuple, centre: np.ndarray, radius: float) -> Optional[tuple]:
    o, d = cam.rays(np.array([uv], float))
    d = d[0]
    oc = o - centre
    b = oc @ d
    c = oc @ oc - radius ** 2
    disc = b * b - c
    if disc < 0:
        return None
    P = o + (-b - np.sqrt(disc)) * d
    N = unit(P - centre)
    return P, unit(d - 2 * (d @ N) * N)


def triangulate(rays: list) -> np.ndarray:
    """Least-squares closest point to several 3D lines [(point, unit dir)]."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in rays:
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ p
    return np.linalg.solve(A, b)


def calibrate_lights(shots: list, cam: Camera, radius_mm: float, min_baseline_deg: float = 10.0) -> list:
    """shots: [{"outline": img, "leds": {index: img}}] (one per ball position). Returns rti_lights entries."""
    rays = {}
    for sh in shots:
        ball = find_ball(sh["outline"], radius_mm, cam)
        if ball is None:
            continue
        C, uvr = ball
        for i, img in sh["leds"].items():
            hl = find_highlight(img, uvr)
            if hl is None:
                continue
            r = reflected_ray(cam, hl, C, radius_mm)
            if r is not None:
                rays.setdefault(int(i), []).append(r)
    nominal = nominal_leds()
    out = []
    for i, rs in sorted(rays.items()):
        angles = [np.degrees(np.arccos(np.clip(a[1] @ b[1], -1, 1))) for a in rs for b in rs]
        if len(rs) >= 2 and max(angles) >= min_baseline_deg:
            pos, src = triangulate(rs), "sphere"
        else:
            P, dirv = rs[0]
            zt = nominal.get(i, np.array([0, 0, 68.0]))[2]
            s = (zt - P[2]) / dirv[2] if abs(dirv[2]) > 1e-3 else 150.0
            pos, src = P + max(s, 20.0) * dirv, "sphere+cad-height"
        resid = float(np.mean([np.linalg.norm(np.cross(pos - p, d)) for p, d in rs]))
        out.append({"index": i, "position_mm": [round(float(x), 2) for x in pos], "direction": unit(pos).round(4).tolist(),
                    "intensity": 1.0, "source": src, "rays": len(rs), "residual_mm": round(resid, 2)})
    return out
