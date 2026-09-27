"""Surface maps from a normal field: integrated relief, curvature, enhanced renders, colour maps."""
from __future__ import annotations

import numpy as np


def frankot_chellappa(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Height (pixel units) whose gradient best matches the normals (image axes: x right, y up)."""
    nzc = np.where(mask, np.maximum(nz, 0.15), 1.0)
    p = np.where(mask, -nx / nzc, 0.0)            # dz/dx
    q = np.where(mask, ny / nzc, 0.0)             # dz/drow (rows go down, y goes up)
    h, w = p.shape
    wx = np.fft.fftfreq(w) * 2 * np.pi
    wy = np.fft.fftfreq(h) * 2 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    P, Q = np.fft.fft2(p), np.fft.fft2(q)
    den = WX ** 2 + WY ** 2
    den[0, 0] = 1.0
    Z = (-1j * WX * P - 1j * WY * Q) / den
    Z[0, 0] = 0
    z = np.real(np.fft.ifft2(Z))
    return np.where(mask, z - np.median(z[mask]) if mask.any() else z, 0.0)


def local_relief(height: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """High-pass of the height: small features (incisions, tool marks) without the overall shape."""
    import cv2
    m = mask.astype(np.float32)
    hb = cv2.GaussianBlur(height.astype(np.float32) * m, (0, 0), sigma_px)
    mb = cv2.GaussianBlur(m, (0, 0), sigma_px)
    return np.where(mask, height - hb / np.maximum(mb, 1e-3), 0.0)


def curvature(nx: np.ndarray, ny: np.ndarray, mask: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
    """Mean curvature proxy = divergence of the normal field (image axes). Grooves < 0, ridges > 0."""
    import cv2
    a = cv2.GaussianBlur(np.where(mask, nx, 0).astype(np.float32), (0, 0), sigma_px)
    b = cv2.GaussianBlur(np.where(mask, ny, 0).astype(np.float32), (0, 0), sigma_px)
    dx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dy = -cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    k = dx + dy
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return np.where(inner, -k, 0.0)


def colormap(x: np.ndarray, mask: np.ndarray, lo_hi=None, cmap: str = "TWILIGHT_SHIFTED") -> np.ndarray:
    """Signed map -> RGB uint8 (black outside the mask), symmetric robust range."""
    import cv2
    v = x[mask]
    r = lo_hi or (float(np.percentile(np.abs(v), 99)) if v.size else 1.0)
    u = np.clip(0.5 + 0.5 * x / max(r, 1e-9), 0, 1)
    img = cv2.applyColorMap((u * 255).astype(np.uint8), getattr(cv2, "COLORMAP_" + cmap))[:, :, ::-1]
    return np.where(mask[..., None], img, 0).astype(np.uint8)


def shade_normals(n_img: np.ndarray, albedo: np.ndarray, light, mask: np.ndarray, spec: float = 0.0,
                  shininess: float = 40.0, diffuse: float = 1.0) -> np.ndarray:
    """Blinn-Phong render from image-space normals (x right, y up, z viewer). albedo (H,W,3) linear."""
    l = np.asarray(light, float)
    l = l / np.linalg.norm(l)
    ndl = np.clip(np.einsum("hwc,c->hw", n_img, l), 0, 1)
    hv = l + np.array([0, 0, 1.0])
    hv /= np.linalg.norm(hv)
    ndh = np.clip(np.einsum("hwc,c->hw", n_img, hv), 0, 1)
    out = diffuse * albedo * ndl[..., None] + spec * (ndh ** shininess)[..., None]
    return np.where(mask[..., None], out, 0.0)


def to_srgb8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (y * 255 + 0.5).astype(np.uint8)


def normals_png(n_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = ((np.clip(n_img, -1, 1) + 1) * 127.5 + 0.5).astype(np.uint8)
    return np.where(mask[..., None], rgb, np.array([128, 128, 255], np.uint8))
