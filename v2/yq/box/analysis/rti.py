"""Reflectance Transformation Imaging: robust near-field photometric stereo + PTM (Malzbender et al. 2001).

Inputs: rti/led1..8.jpg, ambient.jpg, lights.json (or calibrated rti_lights.json). Per pixel the light
direction and 1/r^2 falloff are computed from the real 3D surface point (depth of the reconstructed
mesh seen from the RTI camera), because the LEDs are only ~15 cm away (a distant-light model would
bend all normals). Shadows (too dark), saturated pixels and specular outliers are rejected per pixel.
Outputs: normals, albedo, relief (integrated height, high-pass), curvature, specular enhancement,
relit renders and the web PTM (format documented in docs/box_analysis.md, "yq-ptm-1").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import Camera, unit
from .masks import linear

MAX_WIDTH = 1600
LED_TARGET = np.array([0.0, 0.0, 40.0])     # LEDs are aimed at the platter (CAD "orientado al plato")


def per_pixel_lights(light_pos: np.ndarray, points: np.ndarray, intensity=None, led_cos_power: float = 1.0) -> tuple:
    """points (P,3) -> unit directions (M,P,3) and relative irradiance (M,P) with 1/r^2 and LED emission cosine."""
    d = light_pos[:, None, :] - points[None]
    r = np.linalg.norm(d, axis=2)
    l = d / r[..., None]
    axis = unit(LED_TARGET[None] - light_pos)
    emit = np.clip(np.einsum("mpc,mc->mp", -l, axis), 0.05, 1.0) ** led_cos_power
    E = emit * (150.0 / r) ** 2
    if intensity is not None:
        E = E * np.asarray(intensity, float)[:, None]
    return l, E


def photometric_stereo(I: np.ndarray, L: np.ndarray, E: np.ndarray, valid: np.ndarray, iters: int = 2,
                       spec_k: float = 2.5, n_prior: Optional[np.ndarray] = None, prior_weight: float = 0.05) -> tuple:
    """I (M,P) linear luminance, L (M,P,3) light dirs, E (M,P) irradiance, valid (M,P) bool.

    Solves J = I/E = l . g (g = albedo * normal) by weighted least squares, then re-weights to drop
    shadows/highlights (residuals far above the robust scale). The box LEDs all sit ~68 mm above the
    platter, so on tall objects the vertical normal component is badly conditioned; `n_prior` (normals of
    the 3D model, (P,3)) adds a weak Tikhonov term prior_weight*trace(A)/3*|g - rho0 n_prior|^2 that only
    matters along the ill-conditioned direction (fine horizontal detail still comes from the photos).
    Returns normals (P,3) (NaN where < 3 usable lights and no prior), albedo (P,), used (M,P), cond (P,)."""
    J = I / np.maximum(E, 1e-6)
    w = valid.astype(np.float64)
    P = I.shape[1]
    n = np.full((P, 3), np.nan)
    rho = np.zeros(P)
    cond = np.full(P, np.inf)
    for it in range(iters + 1):
        A = np.einsum("mp,mpi,mpj->pij", w, L, L)
        b = np.einsum("mp,mp,mpi->pi", w, J, L)
        nl = w.sum(axis=0)
        ok = nl >= 3
        if n_prior is not None:
            s = np.clip(np.einsum("mpi,pi->mp", L, n_prior), 0, None)
            rho0 = (w * J * s).sum(axis=0) / np.maximum((w * s * s).sum(axis=0), 1e-9)
            lam = prior_weight * np.trace(A, axis1=1, axis2=2) / 3.0 + 1e-9
            A = A + lam[:, None, None] * np.eye(3)[None]
            b = b + (lam * rho0)[:, None] * n_prior
            ok = nl >= 2
        A = A + np.eye(3)[None] * 1e-12
        ev = np.linalg.eigvalsh(A)
        cond = ev[:, 2] / np.maximum(ev[:, 0], 1e-12)
        g = np.zeros((P, 3))
        if ok.any():
            g[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
        rho = np.linalg.norm(g, axis=1)
        n = np.where(ok[:, None] & (rho[:, None] > 1e-9), g / np.maximum(rho[:, None], 1e-12), np.nan)
        if it == iters:
            break
        pred = rho[None] * np.clip(np.einsum("mpi,pi->mp", L, np.nan_to_num(n)), 0, None)
        res = J - pred
        scale = 1.4826 * np.median(np.abs(np.where(w > 0, res, 0)), axis=0) + 0.02 * rho + 1e-6
        outlier = (res > spec_k * scale[None]) | ((pred <= 1e-6) & (J > 0.1 * rho[None]))
        keep = (w > 0) & ~outlier
        w = np.where(keep.sum(axis=0)[None] >= 3, keep, w > 0).astype(np.float64)
    return n, rho, w > 0, cond


def fit_ptm(Y: np.ndarray, lu: np.ndarray, lv: np.ndarray, valid: Optional[np.ndarray] = None, ridge: float = 1e-3) -> np.ndarray:
    """Biquadratic PTM per pixel: Y = a0 lu^2 + a1 lv^2 + a2 lu lv + a3 lu + a4 lv + a5. Y, lu, lv: (M,P).
    Raking LEDs all at similar elevation make lu^2+lv^2 almost constant, so a small ridge term picks the
    minimum-norm solution (documented limitation of 8-light PTMs)."""
    phi = np.stack([lu * lu, lv * lv, lu * lv, lu, lv, np.ones_like(lu)], axis=-1)       # (M,P,6)
    w = np.ones_like(Y) if valid is None else valid.astype(np.float64)
    A = np.einsum("mp,mpi,mpj->pij", w, phi, phi)
    tr = np.trace(A, axis1=1, axis2=2)[:, None, None] / 6.0
    A = A + ridge * np.maximum(tr, 1e-9) * np.eye(6)[None]
    b = np.einsum("mp,mp,mpi->pi", w, Y, phi)
    return np.linalg.solve(A, b[..., None])[..., 0]


def eval_ptm(c: np.ndarray, lu, lv) -> np.ndarray:
    return c[..., 0] * lu * lu + c[..., 1] * lv * lv + c[..., 2] * lu * lv + c[..., 3] * lu + c[..., 4] * lv + c[..., 5]


def export_web_ptm(coef: np.ndarray, albedo_rgb: np.ndarray, mask: np.ndarray, normals_img: np.ndarray, out_dir: Path,
                   fit_lights: list, crop: list, prefix: str = "rti_") -> dict:
    """Write the 'yq-ptm-1' web format: two coefficient PNGs + albedo RGBA PNG + normals PNG + JSON."""
    import cv2
    from .rti_maps import normals_png, to_srgb8
    out_dir = Path(out_dir)
    lo = np.percentile(coef[mask], 0.5, axis=0) if mask.any() else np.zeros(6)
    hi = np.percentile(coef[mask], 99.5, axis=0) if mask.any() else np.ones(6)
    scale = np.maximum(hi - lo, 1e-6)
    q = np.clip(np.round((coef - lo) / scale * 255.0), 0, 255).astype(np.uint8)
    q[~mask] = 0
    files = {"coef012": prefix + "ptm_c012.png", "coef345": prefix + "ptm_c345.png", "albedo": prefix + "albedo.png",
             "normals": prefix + "normals.png", "json": prefix + "ptm.json"}
    cv2.imwrite(str(out_dir / files["coef012"]), q[..., [2, 1, 0]])
    cv2.imwrite(str(out_dir / files["coef345"]), q[..., [5, 4, 3]])
    rgba = np.concatenate([to_srgb8(albedo_rgb), (mask * 255).astype(np.uint8)[..., None]], axis=2)
    cv2.imwrite(str(out_dir / files["albedo"]), rgba[..., [2, 1, 0, 3]])
    cv2.imwrite(str(out_dir / files["normals"]), normals_png(normals_img, mask)[..., ::-1])
    meta = {"format": "yq-ptm-1", "width": int(mask.shape[1]), "height": int(mask.shape[0]),
            "scale": [float(x) for x in scale], "bias": [float(x) for x in lo],
            "decode": "a[k] = (byte/255) * scale[k] + bias[k]; c012.png R,G,B = a0,a1,a2; c345.png R,G,B = a3,a4,a5",
            "formula": "L = a0*lu*lu + a1*lv*lv + a2*lu*lv + a3*lu + a4*lv + a5",
            "light": "(lu, lv) = unit light direction projected on the image: lu to the right, lv up (|(lu,lv)| <= 1)",
            "color": "rgb_linear = srgb_to_linear(albedo.rgb) * max(L, 0) * gain; alpha = object mask",
            "normals": "n = rgb/255*2-1, x right, y up, z towards the viewer (Blinn-Phong alternative)",
            "gain": 1.0, "fit_lights": [[round(a, 4), round(b, 4)] for a, b in fit_lights], "crop_xywh": crop, "files": files}
    (out_dir / files["json"]).write_text(json.dumps(meta, indent=1))
    return files
