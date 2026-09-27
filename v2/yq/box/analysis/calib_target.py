"""Printable calibration targets with exact millimetre sizes (SVG, PDF and PNG at 600 dpi).

Print at 100 % ("tamaño real", no "ajustar a la página") on A4, then check the 100 mm scale bar with
a ruler or calipers; write the measured length into --measured-mm so the tools correct the scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .calib_charuco import INTRINSICS_BOARD, TURNTABLE_BOARD, BoardSpec
from .calib_thermal import THERMAL_BOARD

A4 = (210.0, 297.0)
DPI = 600


def charuco_bits(spec: BoardSpec, px_per_mm: float) -> np.ndarray:
    """Board image (uint8, white=255) of exactly cols*square x rows*square mm."""
    b = spec.board()
    W, H = spec.size_mm
    return b.generateImage((int(round(W * px_per_mm)), int(round(H * px_per_mm))), marginSize=0, borderBits=1)


def checker_bits(px_per_mm: float) -> np.ndarray:
    cols, rows = THERMAL_BOARD["inner"][0] + 1, THERMAL_BOARD["inner"][1] + 1
    s = int(round(THERMAL_BOARD["square_mm"] * px_per_mm))
    img = np.full((rows * s, cols * s), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[r * s:(r + 1) * s, c * s:(c + 1) * s] = 0
    return img


def _page(board: np.ndarray, title: str, px_per_mm: float) -> np.ndarray:
    import cv2
    pw, ph = int(round(A4[0] * px_per_mm)), int(round(A4[1] * px_per_mm))
    page = np.full((ph, pw), 255, np.uint8)
    y0 = int(35 * px_per_mm)
    x0 = (pw - board.shape[1]) // 2
    page[y0:y0 + board.shape[0], x0:x0 + board.shape[1]] = board
    yb = y0 + board.shape[0] + int(15 * px_per_mm)          # 100 mm scale bar with 10 mm ticks
    xb = (pw - int(100 * px_per_mm)) // 2
    t = max(2, int(0.4 * px_per_mm))
    cv2.rectangle(page, (xb, yb), (xb + int(100 * px_per_mm), yb + t), 0, -1)
    for k in range(11):
        x = xb + int(k * 10 * px_per_mm)
        cv2.rectangle(page, (x, yb - int(3 * px_per_mm)), (x + t, yb + t), 0, -1)
    fs = px_per_mm * 0.12
    cv2.putText(page, title, (int(15 * px_per_mm), int(20 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, fs, 0, max(1, int(fs * 2)))
    cv2.putText(page, "Imprimir al 100 % (tamano real). La barra debe medir 100 mm.", (int(15 * px_per_mm), yb + int(12 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, 0, max(1, int(fs * 1.4)))
    return page


def svg_from_bits(bits: np.ndarray, px_per_mm: float, title: str) -> str:
    """Exact vector SVG: one rect per horizontal run of black pixels, in millimetres."""
    w_mm, h_mm = A4
    y_off = 35.0
    x_off = (w_mm - bits.shape[1] / px_per_mm) / 2
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%gmm" height="%gmm" viewBox="0 0 %g %g">' % (w_mm, h_mm, w_mm, h_mm),
             '<rect width="100%" height="100%" fill="white"/>', '<g fill="black" shape-rendering="crispEdges">']
    blk = bits < 128
    for r in range(blk.shape[0]):
        row = blk[r]
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            parts.append('<rect x="%.4f" y="%.4f" width="%.4f" height="%.4f"/>' % (
                x_off + s / px_per_mm, y_off + r / px_per_mm, (e - s) / px_per_mm, 1.0 / px_per_mm))
    yb = y_off + bits.shape[0] / px_per_mm + 15
    parts.append('<rect x="55" y="%.3f" width="100" height="0.4"/>' % yb)
    parts += ['<rect x="%g" y="%.3f" width="0.4" height="3.4"/>' % (55 + 10 * k, yb - 3) for k in range(11)]
    parts.append('</g><text x="15" y="20" font-size="6" font-family="sans-serif">%s</text>' % title)
    parts.append('<text x="15" y="%.1f" font-size="4" font-family="sans-serif">Imprimir al 100 %%. La barra mide 100 mm.</text>' % (yb + 12))
    parts.append("</svg>")
    return "\n".join(parts)


def make_targets(out_dir: Path) -> dict:
    """Write intrinsics / turntable / thermal targets as PNG (600 dpi), PDF and SVG."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ppm = DPI / 25.4
    svg_ppm = 4.0                                           # SVG uses 0.25 mm cells: exact for 18/13.5/25 mm geometry
    made = {}
    for name, title, bits_fn in (
            ("charuco_intrinsics", "Yachachiq - ChArUco camaras 8x6, cuadro 18 mm", lambda p: charuco_bits(INTRINSICS_BOARD, p)),
            ("charuco_plato", "Yachachiq - ChArUco plato 6x6, cuadro 18 mm", lambda p: charuco_bits(TURNTABLE_BOARD, p)),
            ("termico_ajedrez", "Yachachiq - tablero termico 6x5, cuadro 25 mm", checker_bits)):
        page = _page(bits_fn(ppm), title, ppm)
        png = out_dir / (name + ".png")
        im = Image.fromarray(page)
        im.save(png, dpi=(DPI, DPI))
        im.convert("RGB").save(out_dir / (name + ".pdf"), resolution=DPI)
        (out_dir / (name + ".svg")).write_text(svg_from_bits(bits_fn(svg_ppm), svg_ppm, title))
        made[name] = [str(png), str(out_dir / (name + ".pdf")), str(out_dir / (name + ".svg"))]
    return made
