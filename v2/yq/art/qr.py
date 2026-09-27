"""QR code drawn by the pen (back page), error correction H.

    story_url(story_id)                          -> config.PUBLIC_BASE_URL + story id
    matrix(data)                                 -> bool array (True = dark module), no quiet zone
    strokes(data, x, y, size_mm, pen_mm)         -> serpentine hatch strokes filling the dark modules
    decode_strokes(strokes, ...)                 -> text read back with OpenCV (tests / self-check)

Library: segno (BSD-3-Clause) when installed, otherwise OpenCV's QRCodeEncoder
(Apache-2.0, already in requirements-common). Each row's run of dark modules is
filled with horizontal lines spaced by the pen width and linked into a single
zig-zag stroke, inset by half a pen width so the ink does not bleed into the
light modules.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from yq.common import config

from . import settings
from .geom import Stroke
from .hatch import hatch_rect


def story_url(story_id: str, published: bool = True) -> str:
    """The story's page when it was published, else the general gallery."""
    if not published or not story_id:
        return config.PUBLIC_BASE_URL
    return settings.QR_URL_FORMAT.format(base=config.PUBLIC_BASE_URL, story_id=story_id)


def matrix(data: str) -> np.ndarray:
    try:
        import segno
        q = segno.make_qr(data, error=str(settings.QR_ERROR).lower(), boost_error=False)
        m = np.array([[bool(v) for v in row] for row in q.matrix], dtype=bool)
        return m
    except ImportError:
        pass
    import cv2
    p = cv2.QRCodeEncoder_Params()
    p.correction_level = {"l": cv2.QRCodeEncoder_CORRECT_LEVEL_L, "m": cv2.QRCodeEncoder_CORRECT_LEVEL_M,
                          "q": cv2.QRCodeEncoder_CORRECT_LEVEL_Q,
                          "h": cv2.QRCodeEncoder_CORRECT_LEVEL_H}[str(settings.QR_ERROR).lower()]
    enc = cv2.QRCodeEncoder.create(p)
    img = enc.encode(data)                         # uint8, 0 = dark, includes a quiet zone
    dark = img < 128
    ys, xs = np.nonzero(dark)
    return dark[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def strokes(data: str, x: float, y: float, size_mm: float = None, pen_mm: float = None,
            overlap: float = 0.85) -> Tuple[List[Stroke], dict]:
    """Strokes (mm) for the QR symbol with its top-left corner at (x, y); the quiet zone
    (4 modules) is OUTSIDE this square and must be left blank by the layout.
    overlap: hatch spacing as a fraction of the pen width (< 1 = lines overlap, solid black)."""
    size_mm = settings.QR_SIZE_MM if size_mm is None else size_mm
    pen = settings.PEN_WIDTH_MM if pen_mm is None else pen_mm
    m = matrix(data)
    n = m.shape[0]
    mod = size_mm / float(n)
    out: List[Stroke] = []
    spacing = pen * overlap
    for r in range(n):
        row = m[r]
        c = 0
        while c < n:
            if not row[c]:
                c += 1
                continue
            c2 = c
            while c2 + 1 < n and row[c2 + 1]:
                c2 += 1
            out.append(hatch_rect(x + c * mod, y + r * mod, x + (c2 + 1) * mod, y + (r + 1) * mod,
                                  spacing, pen))
            c = c2 + 1
    info = {"modules": n, "module_mm": round(mod, 3), "quiet_mm": round(4 * mod, 2), "version": (n - 17) // 4}
    return out, info


def rasterize(strokes_mm: List[Stroke], x: float, y: float, size_mm: float, pen_mm: float,
              px_per_mm: float = 12.0, quiet_mm: float = 8.0) -> np.ndarray:
    """Render strokes at the real pen width (round caps) on white, with the quiet zone."""
    from .geom import render
    W = int(math.ceil((size_mm + 2 * quiet_mm) * px_per_mm))
    return render(strokes_mm, W, W, width_px=max(1.0, pen_mm * px_per_mm), scale=px_per_mm,
                  dx=(quiet_mm - x) * px_per_mm, dy=(quiet_mm - y) * px_per_mm)


def decode_strokes(strokes_mm: List[Stroke], x: float, y: float, size_mm: float, pen_mm: float,
                   px_per_mm: float = 12.0) -> str:
    import cv2
    img = rasterize(strokes_mm, x, y, size_mm, pen_mm, px_per_mm)
    det = cv2.QRCodeDetector()
    text, pts, _ = det.detectAndDecode(img)
    if not text:
        small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        text, pts, _ = det.detectAndDecode(small)
    return text or ""
