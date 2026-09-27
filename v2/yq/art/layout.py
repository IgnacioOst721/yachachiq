"""Page layout in paper millimetres (origin top-left, y down).

    front_page(trace, title="")          -> strokes: the drawing fitted into the printable area
    back_page(story, title, qr_url)      -> (strokes, info): title, story text (original + Spanish
                                            translation when different), credit line, QR code
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from . import qr, settings, text
from .geom import Stroke, clip_to_box, transform


def fit_box(w: float, h: float, box: tuple, align_top: bool = False) -> Tuple[float, float, float]:
    """(scale, dx, dy) that fits a w x h picture into box = (x0, y0, x1, y1), centred."""
    x0, y0, x1, y1 = box
    s = min((x1 - x0) / w, (y1 - y0) / h)
    dx = x0 + ((x1 - x0) - w * s) / 2.0
    dy = y0 if align_top else y0 + ((y1 - y0) - h * s) / 2.0
    return s, dx, dy


def front_box(with_title: bool = None) -> tuple:
    x0, y0, x1, y1 = settings.printable_box()
    if settings.FRONT_TITLE if with_title is None else with_title:
        y1 -= 14.0
    return (x0, y0, x1, y1)


def front_page(strokes_px: List[Stroke], w: int, h: int, title: str = "") -> Tuple[List[Stroke], dict]:
    box = front_box(bool(title) and settings.FRONT_TITLE)
    s, dx, dy = fit_box(w, h, box)
    out = clip_to_box(transform(strokes_px, s, dx, dy), *settings.printable_box())
    info = {"mm_per_px": round(s, 4), "box": [round(v, 2) for v in box]}
    if title and settings.FRONT_TITLE:
        x0, _, x1, y1 = settings.printable_box()
        size = 5.0
        while text.width(title, size, settings.TITLE_FONT) > (x1 - x0) and size > 3.0:
            size -= 0.25
        tw = text.width(title, size, settings.TITLE_FONT)
        out += text.render_line(title, x0 + ((x1 - x0) - tw) / 2.0, y1 - 4.0, size, settings.TITLE_FONT)
    return out, info


def _language_name_es(code: str) -> str:
    try:
        from yq.common import languages
        lang = languages.get(code)
        if lang is not None:
            return getattr(lang, "name_es", "") or code
    except Exception:
        pass
    return {"quy_Latn": "quechua", "quz_Latn": "quechua", "eng_Latn": "inglés", "spa_Latn": "castellano",
            "ayr_Latn": "aimara", "por_Latn": "portugués", "fra_Latn": "francés"}.get(code, code)


def back_page(story_text: str, lang: str, text_es: str = "", title: str = "", qr_url: str = "",
              credit: str = None, date: str = None) -> Tuple[List[Stroke], dict]:
    x0, y0, x1, y1 = settings.printable_box()
    W = x1 - x0
    out: List[Stroke] = []
    info: dict = {}
    y = y0 + 2.0
    # --- title -------------------------------------------------------------------------
    if title:
        st, size, h, _ = text.fit_paragraph(title, x0, y, W, 24.0, 8.0, 5.0, settings.TITLE_FONT,
                                            line_height=1.5, align="center")
        out += st
        y += h + 3.0
        out.append(np.array([[x0 + W * 0.3, y], [x0 + W * 0.7, y]]))
        y += 6.0
    # --- footer: QR bottom-right, caption + credit to its left -----------------------------
    qsize = settings.QR_SIZE_MM
    foot_top = y1 - qsize
    if qr_url:
        qs, qinfo = qr.strokes(qr_url, x1 - qsize, foot_top, qsize, settings.PEN_WIDTH_MM)
        out += qs
        info["qr"] = qinfo
        foot_top -= qinfo["quiet_mm"]
    cap_w = W - qsize - 2 * info.get("qr", {}).get("quiet_mm", 3.0) - 2.0
    cy = y1 - 16.0
    if qr_url:
        out += text.paragraph("Escanea el código para ver tu historia en la web", x0, cy - 12.0, cap_w, 2.8,
                              "futural")[0]
    credit = settings.CREDIT if credit is None else credit
    date = date or time.strftime("%d/%m/%Y")
    out += text.paragraph("%s - %s" % (credit, date), x0, cy, cap_w, 2.2, "futural")[0]
    # --- body ------------------------------------------------------------------------------
    body_top, body_bot = y, foot_top - 6.0
    body_h = body_bot - body_top
    orig = " ".join((story_text or "").split())
    es = " ".join((text_es or "").split())
    show_orig = bool(orig) and text.supported_fraction(orig) >= 0.9
    show_es = bool(es) and (es.lower() != orig.lower()) and not lang.startswith("spa")
    parts: List[Tuple[str, str]] = []
    if show_orig:
        parts.append(("", orig))
    elif orig:
        parts.append(("(Historia contada en %s)" % _language_name_es(lang), ""))
    if show_es:
        parts.append(("En castellano:" if show_orig else "", es))
    if not parts and es:
        parts.append(("", es))
    size = settings.TEXT_MAX_MM
    gap_mm = 6.0

    def layout(sz: float, heights_cap: Optional[List[float]] = None):
        strokes_, yy, fitted = [], body_top, True
        for k, (label, body) in enumerate(parts):
            if label:
                ls, lh, _ = text.paragraph(label, x0, yy, W, sz * 0.85, "rowmans")
                strokes_ += ls
                yy += lh + sz * 0.6
            if body:
                if heights_cap is None:
                    bs, bh, _ = text.paragraph(body, x0, yy, W, sz, settings.TEXT_FONT, align="justify")
                else:
                    bs, _, bh, ok = text.fit_paragraph(body, x0, yy, W, heights_cap[k], sz, sz,
                                                       settings.TEXT_FONT, align="justify")
                    fitted = fitted and ok
                strokes_ += bs
                yy += bh
            yy += gap_mm
        return strokes_, yy - gap_mm - body_top, fitted

    while True:
        body_strokes, used, fitted = layout(size)
        if used <= body_h or size <= settings.TEXT_MIN_MM:
            break
        size = round(size - 0.1, 3)
    if used > body_h:                                  # still too long at the minimum size: cut
        per = [max(10.0, (body_h - gap_mm * (len(parts) - 1) - 8.0) / len(parts))] * len(parts)
        body_strokes, used, fitted = layout(size, per)
        info["truncated"] = True
    out += body_strokes
    info.update({"text_mm": size, "parts": len(parts), "body_mm": round(used, 1)})
    return out, info
