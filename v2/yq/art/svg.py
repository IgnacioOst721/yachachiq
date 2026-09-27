"""Plotter SVG (CONTRACTS.md §8): millimetre units, one <path> per stroke, one group per pen.

    to_svg({"black": strokes, ...}, paper_w, paper_h, pen_width) -> SVG text
    preview_png(layers, path, ...)                               -> PNG preview for the kiosk
Coordinates are paper millimetres, origin top-left, y down (the SVG convention).
Groups carry Inkscape layer attributes so vpype/AxiDraw/Inkscape see one layer per pen.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
from xml.sax.saxutils import escape

import numpy as np

from . import settings
from .geom import Stroke

_COLORS = {"black": "#000000", "blue": "#1f3fbf", "red": "#c0282d", "green": "#1e7d32", "brown": "#6b3e1f"}


def _d(s: Stroke) -> str:
    if len(s) == 1:
        x, y = s[0]
        return "M%.2f %.2fL%.2f %.2f" % (x, y, x + 0.01, y)
    parts = ["M%.2f %.2f" % (s[0][0], s[0][1])]
    parts += ["L%.2f %.2f" % (x, y) for x, y in s[1:]]
    return "".join(parts)


def to_svg(layers: Dict[str, Sequence[Stroke]], paper_w: float = None, paper_h: float = None,
           pen_width: float = None, title: str = "", background: bool = False) -> str:
    W = settings.PAPER_W_MM if paper_w is None else paper_w
    H = settings.PAPER_H_MM if paper_h is None else paper_h
    pw = settings.PEN_WIDTH_MM if pen_width is None else pen_width
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" version="1.1" '
           'width="%gmm" height="%gmm" viewBox="0 0 %g %g">' % (W, H, W, H)]
    if title:
        out.append("<title>%s</title>" % escape(title))
    if background:
        out.append('<rect x="0" y="0" width="%g" height="%g" fill="#ffffff"/>' % (W, H))
    for i, (pen, strokes) in enumerate(layers.items()):
        color = _COLORS.get(pen, "#000000")
        out.append('<g id="pen%d" inkscape:groupmode="layer" inkscape:label="%d %s" fill="none" stroke="%s" '
                   'stroke-width="%g" stroke-linecap="round" stroke-linejoin="round">'
                   % (i + 1, i + 1, escape(pen), color, pw))
        for s in strokes:
            if len(s):
                out.append('<path d="%s"/>' % _d(np.asarray(s)))
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def preview_png(layers: Dict[str, Sequence[Stroke]], path, paper_w: float = None, paper_h: float = None,
                pen_width: float = None, px_per_mm: float = None, max_bytes: int = 0) -> str:
    """Render the page as the pen would draw it (grey paper edge, ink at the pen width)."""
    import cv2
    from .geom import render
    W = settings.PAPER_W_MM if paper_w is None else paper_w
    H = settings.PAPER_H_MM if paper_h is None else paper_h
    pw = settings.PEN_WIDTH_MM if pen_width is None else pen_width
    k = settings.PREVIEW_PX_PER_MM if px_per_mm is None else px_per_mm
    img = np.full((int(round(H * k)), int(round(W * k))), 255, np.uint8)
    for strokes in layers.values():
        render(list(strokes), img.shape[1], img.shape[0], width_px=max(1.0, pw * k), scale=k, canvas=img)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), 200, 1)
    params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    cv2.imwrite(str(path), img, params)
    if max_bytes:
        import os
        while os.path.getsize(path) > max_bytes and k > 1.0:
            k *= 0.8
            small = cv2.resize(img, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(path), small, params)
    return str(path)
