"""ART settings: paper, pen, plotter and drawing-pipeline knobs.

Every value can be changed without touching code with an environment variable
YQ_ART_<NAME> (example: YQ_ART_PAPER_W_MM=148 YQ_ART_PAPER_H_MM=210 for A5).
All lengths are millimetres. Joaquín's printer: set the paper, the pen width,
the pen-lift mode and the feeds here (see v2/docs/art.md, "Cambiar papel o lápiz").
"""
from __future__ import annotations

from yq.common.config import env


def _e(name: str, default):
    return env("ART_" + name, default)


# --- Paper and pen ---------------------------------------------------------------
PAPER_W_MM = _e("PAPER_W_MM", 210.0)        # A4 portrait
PAPER_H_MM = _e("PAPER_H_MM", 297.0)
MARGIN_MM = _e("MARGIN_MM", 10.0)           # nothing is drawn closer than this to the paper edge
PEN_WIDTH_MM = _e("PEN_WIDTH_MM", 0.5)      # line width of the pen (hatching spacing, QR fill, previews)
PENS = _e("PENS", ["black"])                # pen names, first = default (multi-pen ready)

# --- Plotter (GRBL-style G-code) -----------------------------------------------------
PEN_MODE = _e("PEN_MODE", "z")              # "z" (Z axis up/down) | "servo" (M3 S<angle>) | "none" (pen never lifts)
PEN_UP_Z = _e("PEN_UP_Z", 2.0)             # lift height: as low as your pen allows (every lift costs time)
PEN_DOWN_Z = _e("PEN_DOWN_Z", 0.0)
SERVO_UP = _e("SERVO_UP", 90)               # M3 S value, PEN_MODE = servo
SERVO_DOWN = _e("SERVO_DOWN", 30)
SERVO_DELAY_S = _e("SERVO_DELAY_S", 0.15)   # G4 dwell after every servo move
DRAW_FEED = _e("DRAW_FEED", 1500.0)         # mm/min while drawing
TRAVEL_FEED = _e("TRAVEL_FEED", 3000.0)     # mm/min pen up (G0 ignores F on GRBL; used for the time estimate)
PEN_FEED = _e("PEN_FEED", 2000.0)           # mm/min for Z moves
ACCEL_MM_S2 = _e("ACCEL_MM_S2", 300.0)      # machine acceleration, only for the time estimate
Y_UP = _e("Y_UP", True)                     # G-code origin at the bottom-left corner (y grows up), like v1
SWAP_XY = _e("SWAP_XY", False)              # machine X runs along the paper height
ORIGIN_X_MM = _e("ORIGIN_X_MM", 0.0)        # offset of the paper corner in machine coordinates
ORIGIN_Y_MM = _e("ORIGIN_Y_MM", 0.0)
MACHINE_W_MM = _e("MACHINE_W_MM", 0.0)      # machine travel limits (0 = same as the paper)
MACHINE_H_MM = _e("MACHINE_H_MM", 0.0)
CONTINUOUS = _e("CONTINUOUS", False)        # force one continuous line (automatic when PEN_MODE = none)

# --- Front drawing -------------------------------------------------------------------
IMAGE_W = _e("IMAGE_W", 768)                # generator resolution (multiple of 16), portrait like A4
IMAGE_H = _e("IMAGE_H", 1088)
WORK_PX = _e("WORK_PX", 1400)               # vectorizer working resolution (longest side, >= 1024)
MIN_STROKE_MM = _e("MIN_STROKE_MM", 1.0)    # shorter strokes are dropped unless they are an isolated detail (eye, dot)
SIMPLIFY_PX = _e("SIMPLIFY_PX", 0.35)       # Douglas-Peucker tolerance (sub-pixel)
SMOOTH_SIGMA_PX = _e("SMOOTH_SIGMA_PX", 1.2)
FILL_STYLE = _e("FILL_STYLE", "hatch")      # filled black areas: "outline" | "hatch" (outline + hatching) | "solid"
HATCH_SPACING_MM = _e("HATCH_SPACING_MM", 1.1)
HATCH_ANGLE_DEG = _e("HATCH_ANGLE_DEG", 45.0)
TONES = _e("TONES", False)                  # engraving-style hatching of grey areas (richer, slower)
DROP_FRAME = _e("DROP_FRAME", True)         # remove borders/frames the generator adds
MAX_DRAW_MINUTES = _e("MAX_DRAW_MINUTES", 25.0)   # simplify harder if the estimate is longer than this
FRONT_TITLE = _e("FRONT_TITLE", False)      # also write the title under the drawing on the front

# --- Back page -----------------------------------------------------------------------
QR_SIZE_MM = _e("QR_SIZE_MM", 36.0)         # QR symbol side without the quiet zone (>= 30)
QR_ERROR = _e("QR_ERROR", "m")              # l / m / q / h. M (15 %) is plenty for a clean pen drawing and
                                            # gives a smaller symbol (bigger modules, fewer strokes) than H
QR_URL_FORMAT = _e("QR_URL_FORMAT", "{base}{story_id}/")  # base = config.PUBLIC_BASE_URL (publish writes <id>/index.html)
TEXT_FONT = _e("TEXT_FONT", "futural")      # single-stroke Hershey font for the story text
TITLE_FONT = _e("TITLE_FONT", "scripts")
TEXT_MAX_MM = _e("TEXT_MAX_MM", 5.0)        # cap height of the story text (shrinks to fit)
TEXT_MIN_MM = _e("TEXT_MIN_MM", 2.4)
CREDIT = _e("CREDIT", "Dibujado por Yachachiq - Colegio FDR, Lima - WRO 2026")

# --- Mac pipeline ----------------------------------------------------------------------
MAX_ATTEMPTS = _e("MAX_ATTEMPTS", 3)        # image generations per story (verify + regenerate)
PREVIEW_PX_PER_MM = _e("PREVIEW_PX_PER_MM", 4.0)


def printable_box() -> tuple:
    """(x0, y0, x1, y1) in mm, SVG convention (origin top-left, y grows down)."""
    return (MARGIN_MM, MARGIN_MM, PAPER_W_MM - MARGIN_MM, PAPER_H_MM - MARGIN_MM)


def continuous() -> bool:
    return bool(CONTINUOUS) or PEN_MODE == "none"
