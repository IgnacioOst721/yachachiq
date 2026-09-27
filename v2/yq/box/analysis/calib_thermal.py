"""Thermal <-> visible registration with a heated printed checkerboard (optional).

Print the thermal target (calib_target.py, 25 mm squares, black ink on paper glued to cardboard),
warm it 20-30 s with the halogen: the black squares get hotter and the Lepton sees the pattern.
Take one Lepton frame and one camera-A photo of the target at the same platter angle.
Result: homography thermal px -> camera px (valid for surfaces near that plane, used for overlays)
and the thermal camera pose (solvePnP with the Lepton intrinsics) in the object frame.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import Camera, intrinsics_from_fov, rodrigues, rotvec

THERMAL_BOARD = {"inner": (5, 4), "square_mm": 25.0}      # 6 x 5 squares


def board_points(inner=THERMAL_BOARD["inner"], square=THERMAL_BOARD["square_mm"]) -> np.ndarray:
    cols, rows = inner
    g = np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), -1).reshape(-1, 2).astype(np.float64)
    return np.concatenate([g * square, np.zeros((len(g), 1))], 1)


def find_corners(img: np.ndarray, inner=THERMAL_BOARD["inner"], upsample: int = 1) -> Optional[np.ndarray]:
    """Chessboard corners (N,2) in the ORIGINAL pixel coordinates, or None. Thermal frames (float degC)
    are normalised to 8 bit and upsampled first (160x120 is too small for the detector)."""
    import cv2
    x = img.astype(np.float32)
    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(x, [1, 99])
    g = np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    if upsample > 1:
        g = cv2.resize(g, (g.shape[1] * upsample, g.shape[0] * upsample), interpolation=cv2.INTER_CUBIC)
    ok, c = cv2.findChessboardCornersSB(g, inner, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        g = 255 - g                                       # hot squares may appear white instead of black
        ok, c = cv2.findChessboardCornersSB(g, inner, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if not ok:
            return None
    c = c.reshape(-1, 2).astype(np.float64)
    return (c + 0.5) / upsample - 0.5 if upsample > 1 else c


def register(thermal_frame: np.ndarray, visible: np.ndarray, cam_vis: Camera, platter_deg: float = 0.0,
             direction: int = 1, thermal_K: Optional[np.ndarray] = None, to_camera: str = "A") -> dict:
    """thermal_frame (120,160) degC, visible RGB photo of camera `to_camera` (cam_vis at platter 0)."""
    import cv2
    tc = find_corners(thermal_frame, upsample=4)
    vc = find_corners(visible)
    if tc is None or vc is None:
        raise ValueError("no se detectó el tablero térmico en %s" % ("la imagen térmica" if tc is None else "la foto"))
    # the detector may start the corner order at the opposite end in one of the images: align by direction
    if np.dot(tc[-1] - tc[0], (vc[-1] - vc[0]) * [1, 1]) < 0:
        tc = tc[::-1].copy()
    H, _ = cv2.findHomography(tc, vc, 0)
    proj = cv2.perspectiveTransform(tc.reshape(-1, 1, 2), H).reshape(-1, 2)
    rms_h = float(np.sqrt(np.mean(np.sum((proj - vc) ** 2, axis=1))))
    h, w = thermal_frame.shape[:2]
    K = np.asarray(thermal_K) if thermal_K is not None else intrinsics_from_fov(w, h, 57.0)
    obj = board_points()
    view = cam_vis.scaled(visible.shape[1], visible.shape[0]).at_platter(platter_deg, direction)
    okv, rv_v, tv_v = cv2.solvePnP(obj, vc, view.K, view.dist)
    okt, rv_t, tv_t = cv2.solvePnP(obj, tc, K, np.zeros(5))
    if not (okv and okt):
        raise ValueError("solvePnP falló")
    Rvb, tvb = rodrigues(rv_v.ravel()), tv_v.ravel()          # board -> camera(view)
    Rtb, ttb = rodrigues(rv_t.ravel()), tv_t.ravel()          # board -> thermal
    Rbo = Rvb.T @ view.R                                      # object -> board
    tbo = Rvb.T @ (view.t - tvb)
    R = Rtb @ Rbo
    t = Rtb @ tbo + ttb
    pt, _ = cv2.projectPoints(obj, rv_t, tv_t, K, np.zeros(5))
    rms_t = float(np.sqrt(np.mean(np.sum((pt.reshape(-1, 2) - tc) ** 2, axis=1))))
    return {"K": K.tolist(), "dist": [0.0] * 5, "width": w, "height": h, "R": R.tolist(), "t": t.tolist(), "rms_px": rms_t,
            "homography": {"to_camera": to_camera, "platter_deg": platter_deg, "H": H.tolist(),
                           "image_size": [int(visible.shape[1]), int(visible.shape[0])], "rms_px": rms_h},
            "source": "heated-checkerboard"}
