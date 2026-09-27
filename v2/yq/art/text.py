"""Single-stroke text for the pen: Hershey fonts (public-domain data, vendored in fonts/).

    width(text, size_mm, font)                     -> mm
    render_line(text, x, y, size_mm, font)          -> strokes (mm, baseline at y, y down)
    wrap(text, box_w, size_mm, font)                -> list of lines
    paragraph(text, x, y, box_w, size_mm, ...)      -> (strokes, height_mm)
    fit_paragraph(text, x, y, box_w, box_h, ...)    -> (strokes, size_mm, height_mm, fitted)

size_mm is the CAP height. Accented letters (á é í ó ú ñ ü, capitals, à â ç ...) are
composed from the base glyph plus pen-drawn marks; ¿ and ¡ are the rotated ? and !.
Quechua apostrophes (' ’ ʼ) all map to the straight apostrophe glyph.
"""
from __future__ import annotations

import functools
import math
import unicodedata
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .geom import Stroke

FONT_DIR = Path(__file__).resolve().parent / "fonts"
CAP = 21.0          # Hershey cap height in font units (top -12, baseline +9)
BASE = 9.0
XTOP = -5.0         # top of lowercase x-height
CAPTOP = -12.0

_REPLACE = {"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'", "“": '"',
            "”": '"', "«": '"', "»": '"', "–": "-", "—": "-", "…": "...",
            " ": " ", "\t": " ", "º": "o", "ª": "a"}
_MARKS = {"́": "acute", "̀": "grave", "̂": "circ", "̃": "tilde",
          "̈": "diaer", "̧": "cedil", "̊": "ring", "̌": "caron"}


@functools.lru_cache(maxsize=8)
def load_font(name: str) -> dict:
    """{char: (left, right, [strokes in font units])} for ASCII 32..127."""
    path = FONT_DIR / (name + ".jhf")
    raw = path.read_text(encoding="latin-1")
    glyphs, buf = [], ""
    for line in raw.splitlines():
        buf = line if not buf else buf + line
        n = int(buf[5:8])
        if len(buf) - 8 >= 2 * n:
            glyphs.append(buf)
            buf = ""
    out = {}
    for i, g in enumerate(glyphs):
        n = int(g[5:8])
        data = g[8:8 + 2 * n]
        left, right = ord(data[0]) - 82, ord(data[1]) - 82
        strokes, cur = [], []
        for k in range(2, len(data), 2):
            pair = data[k:k + 2]
            if pair == " R":
                if cur:
                    strokes.append(np.array(cur, dtype=float))
                cur = []
                continue
            cur.append((ord(pair[0]) - 82, ord(pair[1]) - 82))
        if cur:
            strokes.append(np.array(cur, dtype=float))
        out[chr(32 + i)] = (left, right, strokes)
    return out


def _mark(kind: str, cx: float, top: float) -> List[Stroke]:
    """Diacritic strokes centred at cx, sitting just above `top` (font units, y down)."""
    y = top - 2.0
    if kind == "acute":
        return [np.array([[cx - 1.5, y], [cx + 2.0, y - 3.5]])]
    if kind == "grave":
        return [np.array([[cx + 1.5, y], [cx - 2.0, y - 3.5]])]
    if kind == "circ":
        return [np.array([[cx - 3.0, y], [cx, y - 3.0], [cx + 3.0, y]])]
    if kind == "caron":
        return [np.array([[cx - 3.0, y - 3.0], [cx, y], [cx + 3.0, y - 3.0]])]
    if kind == "tilde":
        t = np.linspace(0, 1, 9)
        return [np.stack([cx - 3.5 + 7.0 * t, y - 1.5 - 1.2 * np.sin(2 * np.pi * t)], axis=1)]
    if kind == "diaer":
        return [_dot(cx - 2.5, y - 1.0), _dot(cx + 2.5, y - 1.0)]
    if kind == "ring":
        a = np.linspace(0, 2 * np.pi, 13)
        return [np.stack([cx + 1.6 * np.cos(a), y - 1.6 + 1.6 * np.sin(a)], axis=1)]
    if kind == "cedil":
        return [np.array([[cx, BASE], [cx + 0.5, BASE + 2.0], [cx + 2.0, BASE + 3.0], [cx, BASE + 4.5],
                          [cx - 1.5, BASE + 4.0]])]
    return []


def _dot(x: float, y: float, r: float = 0.6) -> Stroke:
    a = np.linspace(0, 2 * np.pi, 7)
    return np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)


def _glyph(ch: str, font: str) -> Tuple[float, float, List[Stroke]]:
    """(left, right, strokes) for any character, composing accents and inverted marks."""
    f = load_font(font)
    ch = _REPLACE.get(ch, ch)
    if len(ch) > 1:                                   # "..." etc.
        parts = [_glyph(c, font) for c in ch]
        x, strokes = 0.0, []
        for l, r, ss in parts:
            strokes += [s + np.array([x - l, 0.0]) for s in ss]
            x += r - l
        return (0.0, x, strokes)
    if ch in f:
        return f[ch]
    if ch in ("¿", "¡"):                    # ¿ ¡ = ? ! rotated 180 degrees, hanging lower
        l, r, ss = f["?" if ch == "¿" else "!"]
        # x' = -x, y' = 3 - y: the dot goes up to the x-height, the hook hangs below the baseline
        return (-r, -l, [np.stack([-s[:, 0], 3.0 - s[:, 1]], axis=1) for s in ss])
    d = unicodedata.normalize("NFD", ch)
    base, marks = d[0], [_MARKS[m] for m in d[1:] if m in _MARKS]
    if base in f and base != ch:
        l, r, ss = f[base]
        if base in "ij" and marks:                     # dotless i / j under an accent
            ss = [s for s in ss if s[:, 1].max() > XTOP - 0.5]
        ss = list(ss)
        cx = 0.0
        pts = np.concatenate(ss) if ss else np.zeros((1, 2))
        if base in "ij":
            cx = float(pts[:, 0].mean())
        top = CAPTOP if base.isupper() or base.isdigit() else XTOP
        for m in marks:
            ss += _mark(m, cx, top if m != "cedil" else BASE)
        return (l, r, ss)
    if "?" in f:                                        # unknown character: a small open box
        return (-5.0, 5.0, [np.array([[-3, BASE], [-3, XTOP], [3, XTOP], [3, BASE], [-3, BASE]], float)])
    return (0.0, 0.0, [])


def supported_fraction(text: str, font: str = "futural") -> float:
    """Share of non-space characters drawable as real letters (non-Latin scripts -> low)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    f = load_font(font)
    ok = 0
    for c in chars:
        c2 = _REPLACE.get(c, c)
        if c2 in f or c in ("¿", "¡") or (unicodedata.normalize("NFD", c)[0] in f and
                                                    all(m in _MARKS for m in unicodedata.normalize("NFD", c)[1:])):
            ok += 1
    return ok / float(len(chars))


def width(text: str, size_mm: float, font: str = "futural") -> float:
    s = size_mm / CAP
    return sum(r - l for l, r, _ in (_glyph(c, font) for c in text)) * s


def render_line(text: str, x: float, y: float, size_mm: float, font: str = "futural",
                spacing: float = 0.0) -> List[Stroke]:
    """Strokes for one line; (x, y) = left end of the baseline, y grows down (mm)."""
    s = size_mm / CAP
    out: List[Stroke] = []
    cx = x
    for c in text:
        l, r, ss = _glyph(c, font)
        for st in ss:
            out.append(np.stack([cx + (st[:, 0] - l) * s, y + (st[:, 1] - BASE) * s], axis=1))
        cx += (r - l) * s + spacing
    return out


def _clean(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def wrap(text: str, box_w: float, size_mm: float, font: str = "futural") -> List[str]:
    """Greedy word wrap; words longer than the line are split with a hyphen."""
    words = _clean(text).split(" ")
    lines: List[str] = []
    cur = ""
    space = width(" ", size_mm, font)
    for w in words:
        if not w:
            continue
        while width(w, size_mm, font) > box_w:          # very long word: break it
            k = len(w) - 1
            while k > 1 and width(w[:k] + "-", size_mm, font) > box_w:
                k -= 1
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(w[:k] + "-")
            w = w[k:]
        if not cur:
            cur = w
        elif width(cur, size_mm, font) + space + width(w, size_mm, font) <= box_w:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def paragraph(text: str, x: float, y: float, box_w: float, size_mm: float, font: str = "futural",
              line_height: float = 1.75, align: str = "left") -> Tuple[List[Stroke], float, List[str]]:
    """Wrapped text whose first line's cap top is at y. align: left | center | justify.
    Returns (strokes, height_mm, lines)."""
    lines = wrap(text, box_w, size_mm, font)
    out: List[Stroke] = []
    lh = size_mm * line_height
    for i, line in enumerate(lines):
        base = y + size_mm + i * lh
        lw = width(line, size_mm, font)
        if align == "center":
            out += render_line(line, x + (box_w - lw) / 2.0, base, size_mm, font)
        elif align == "justify" and i < len(lines) - 1 and " " in line:
            words = line.split(" ")
            gap = (box_w - sum(width(w, size_mm, font) for w in words)) / (len(words) - 1)
            if gap > 3 * width(" ", size_mm, font):      # too sparse: leave it ragged
                out += render_line(line, x, base, size_mm, font)
            else:
                cx = x
                for w in words:
                    out += render_line(w, cx, base, size_mm, font)
                    cx += width(w, size_mm, font) + gap
        else:
            out += render_line(line, x, base, size_mm, font)
    height = (len(lines) - 1) * lh + size_mm * 1.45 if lines else 0.0   # + descenders
    return out, height, lines


def fit_paragraph(text: str, x: float, y: float, box_w: float, box_h: float, max_mm: float, min_mm: float,
                  font: str = "futural", line_height: float = 1.75, align: str = "left"):
    """Largest size in [min_mm, max_mm] whose wrapped text fits the box. If even min_mm does not
    fit, the text is cut at a word boundary and ends with "...".
    Returns (strokes, size_mm, height_mm, fitted: bool)."""
    size = max_mm
    while size >= min_mm - 1e-9:
        st, h, _ = paragraph(text, x, y, box_w, size, font, line_height, align)
        if h <= box_h:
            return st, size, h, True
        size = round(size - 0.1, 3)
    size = min_mm
    words = _clean(text).split(" ")
    lo, hi = 0, len(words)
    while lo < hi:                                      # longest prefix that fits
        mid = (lo + hi + 1) // 2
        _, h, _ = paragraph(" ".join(words[:mid]) + "...", x, y, box_w, size, font, line_height, align)
        if h <= box_h:
            lo = mid
        else:
            hi = mid - 1
    st, h, _ = paragraph(" ".join(words[:lo]) + "...", x, y, box_w, size, font, line_height, align)
    return st, size, h, False
