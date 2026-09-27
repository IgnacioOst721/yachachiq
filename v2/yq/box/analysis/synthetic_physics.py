"""Physics for the synthetic scans: UV fluorescence images and pulsed thermography sequences."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .geometry import Camera
from .synthetic import Shape, shade, to_uint8, trace


def uv_images(shape: Shape, cam: Camera, zone: Optional[dict] = None, rng=None) -> tuple:
    """uv / visible / dark photos from `cam` at platter 0 with one patch of distinct (bluish-white)
    fluorescence, like a modern adhesive. zone: {"azimuth_deg", "half_width_deg", "z_min", "z_max"}."""
    rng = rng or np.random.default_rng(0)
    dims = shape.dims()
    zone = zone or {"azimuth_deg": 90.0, "half_width_deg": 22.0, "z_min": 0.35 * dims["height"], "z_max": 0.6 * dims["height"]}
    tr = trace(shape, cam, ss=2)

    def in_zone(p):
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        daz = np.abs(((az - zone["azimuth_deg"]) + 180) % 360 - 180)
        return (daz <= zone["half_width_deg"]) & (p[:, 2] >= zone["z_min"]) & (p[:, 2] <= zone["z_max"])

    top = [{"kind": "dir", "dir": (0.1, -0.2, 1.0), "power": 0.6}]
    visible = shade(tr, shape, top, ambient=0.3)

    def uv_albedo(p):
        alb = shape.albedo(p)
        leak = 0.10 * alb[:, 2:3] * np.array([[0.45, 0.15, 1.0]])            # violet LED light reflected
        fl = np.tile([[0.035, 0.03, 0.05]], (len(p), 1))                     # dull base fluorescence
        fl[in_zone(p)] = [0.30, 0.38, 0.55]                                   # bright bluish-white patch
        return leak + fl

    uvimg = shade(tr, shape, [{"kind": "dir", "dir": (0.4, 0.4, 0.8), "power": 0.9}], ambient=0.25,
                  platter_albedo=0.02, wall=0.01, rgb_albedo=uv_albedo)
    dark = np.full_like(visible, 0.002)
    hit = tr["hit"] & in_zone(np.where(tr["hit"][:, None], tr["p"], 0))
    H, W = tr["shape"]
    ss = tr["ss"]
    zmask = hit.reshape(H, W).reshape(H // ss, ss, W // ss, ss).mean(axis=(1, 3)) > 0.5
    ims = {"uv": to_uint8(uvimg, noise=0.004, rng=rng), "visible": to_uint8(visible, noise=0.004, rng=rng),
           "dark": to_uint8(dark, noise=0.002, rng=rng)}
    return ims, {"zone": zone, "pixels": int(zmask.sum()), "mask_rle_bbox": _bbox(zmask), "_mask": zmask.tolist()}


def _bbox(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None


def simulate_thermal(shape: Shape, camT: Camera, defect: Optional[dict] = None, rng=None, heat_s: float = 10.0,
                     cool_s: float = 50.0, pre_s: float = 3.0, fps: float = 8.7, ambient: float = 22.0) -> tuple:
    """Long-pulse active thermography of `shape` seen by the thermal camera at platter 0.

    Every object pixel is a 1D wall (6 mm, diffusivity 0.45 mm^2/s, fired clay) heated at the surface;
    inside the defect disk an air gap (adiabatic) sits at `depth_mm`. Heating is deliberately
    non-uniform (halogen spot) and the sequence has Lepton-like artefacts: 50 mK noise, repeated
    frames and one flat-field-correction freeze followed by a small offset jump.
    Returns (sequence (T,120,160) float32 degC, times (T,), meta dict, defect truth dict)."""
    rng = rng or np.random.default_rng(0)
    tr = trace(shape, camT, ss=2)
    H, W = camT.height, camT.width
    frac = tr["hit"].reshape(H * 2, W * 2).reshape(H, 2, W, 2).mean(axis=(1, 3))
    obj = frac > 0.5
    ys, xs = np.nonzero(obj)
    cy, cx = ys.mean(), xs.mean()
    defect = dict(defect or {})
    defect.setdefault("u", float(cx + 0.18 * (xs.max() - xs.min())))
    defect.setdefault("v", float(cy - 0.1 * (ys.max() - ys.min())))
    defect.setdefault("radius_px", 6.0)
    defect.setdefault("depth_mm", 1.0)
    vv, uu = np.mgrid[0:H, 0:W]
    ddisk = obj & ((uu - defect["u"]) ** 2 + (vv - defect["v"]) ** 2 <= defect["radius_px"] ** 2)
    # heat flux map: halogen spot from above, 30 % non-uniform
    spot = np.exp(-(((uu - cx) / (0.9 * W)) ** 2 + ((vv - cy + 0.3 * H) / (0.9 * H)) ** 2))
    S = np.where(obj, 1.4 * (0.7 + 0.3 * spot), 0.0)          # surface heating rate, degC/s at the first node
    dz, nz, alpha, dt = 0.25, 24, 0.45, 0.05
    fo = alpha * dt / dz ** 2
    last = np.full(obj.shape, nz - 1)
    last[ddisk] = int(round(defect["depth_mm"] / dz)) - 1
    Tz = np.zeros((nz,) + obj.shape)
    kidx = np.arange(nz)[:, None, None]
    active = kidx <= last[None]
    hconv = 0.004                                              # surface loss, 1/s
    times = np.arange(-pre_s, heat_s + cool_s, 1.0 / fps)
    frames, t_sim, ti = [], -pre_s, 0
    while ti < len(times):
        if t_sim >= times[ti] - 1e-9:
            frames.append(Tz[0].copy())
            ti += 1
            continue
        lap = np.zeros_like(Tz)
        lap[1:-1] = Tz[2:] - 2 * Tz[1:-1] + Tz[:-2]
        lap[0] = 2 * (Tz[1] - Tz[0])
        lap[-1] = 2 * (Tz[-2] - Tz[-1])
        # adiabatic at each pixel's last active node: mirror condition
        edge = (kidx == last[None]) & (kidx > 0)
        lap = np.where(edge, 2 * (np.roll(Tz, 1, axis=0) - Tz), lap)
        Tz = np.where(active, Tz + fo * lap, 0.0)
        heating = 1.0 if 0.0 <= t_sim < heat_s else 0.0
        Tz[0] += dt * (heating * S * (2.0 / 1.0) - hconv * Tz[0])
        t_sim += dt
    import cv2
    seq = np.stack([cv2.GaussianBlur(f.astype(np.float32), (0, 0), 0.8) for f in frames])
    wall = 0.25 * np.clip(times / heat_s, 0, 1)[:, None, None] * (~obj)[None]
    seq = ambient + seq + wall + rng.normal(0, 0.05, seq.shape)
    # Lepton artefacts: repeated frames and a 5-frame FFC freeze + 0.15 degC offset afterwards
    for k in rng.choice(np.arange(5, len(seq)), size=max(1, len(seq) // 20), replace=False):
        seq[k] = seq[k - 1]
    kf = int(np.searchsorted(times, heat_s + 20.0))
    seq[kf:kf + 5] = seq[kf - 1]
    seq[kf + 5:] += 0.15
    meta = {"heat_on_s": 0.0, "heat_off_s": heat_s, "halogen": "MR16 35W", "ambient_c": ambient, "fps": fps,
            "camera": "T", "platter_deg": 0.0, "synthetic": True}
    truth = {"u": defect["u"], "v": defect["v"], "radius_px": defect["radius_px"], "depth_mm": defect["depth_mm"],
             "pixels": int(ddisk.sum()), "object_pixels": int(obj.sum())}
    return seq.astype(np.float32), times.astype(np.float64), meta, truth
