"""Check a generated picture before it goes to the pen.

    metrics(image) -> dict            cheap pixel measurements (CPU): ink, grey, colour, dark blobs
    verify(image, elements) -> dict   VLM: are the must-appear elements there? line art? text? frame?
    score(v) -> float 0..1            used to pick the best attempt

The VLM answers per element (true/false) plus style questions; the pixel metrics catch
what a VLM tends to miss (large black fills, grey shading, colour). Problems are named so
the retry prompt can fix them (art_style.emphasize).
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from yq.common import config

log = logging.getLogger("yq.art.verify")

VERIFY_PROMPT = """You check a drawing made by a robot for a visitor's story.
Items that must be visible:
{items}

For EACH item answer true only if it is clearly drawn and recognisable (a vague shape is false).
When an item has a description after the colon, it is true ONLY if the drawing matches that description.
Then judge the style: the goal is clean black ink line art on a plain white background.
Reply with JSON exactly like:
{{"elements": {{{example}}},
 "line_art": 0-10 (10 = clean black outlines on white; 0 = photo, painting, colour or heavy shading),
 "text": true if there are letters, words, numbers or a signature anywhere,
 "frame": true if there is a border or frame around the picture,
 "note": "one short sentence about the biggest problem"}}"""


def metrics(image) -> dict:
    """Pixel statistics of the picture (any size)."""
    import cv2
    rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    small = cv2.resize(rgb, (384, int(384 * rgb.shape[0] / rgb.shape[1])), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ink = gray < 0.5
    colour = (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)
    greyish = (gray > 0.25) & (gray < 0.8)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150) > 0
    near_edge = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    flat_grey = greyish & ~near_edge                     # grey that is not just anti-aliasing
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    blobs = dist > 4.0                                   # ink thicker than ~8 px at 384 px wide
    border = np.zeros_like(ink)
    b = 6
    border[:b, :] = border[-b:, :] = True
    border[:, :b] = border[:, -b:] = True
    return {"ink": round(float(ink.mean()), 4), "colour": round(float(colour.mean()), 4),
            "grey": round(float(flat_grey.mean()), 4), "black_fill": round(float(blobs.mean()), 4),
            "border_ink": round(float(ink[border].mean()), 4)}


def problems_from(m: dict, v: dict = None) -> List[str]:
    p = []
    if m["ink"] > 0.30 or m["black_fill"] > 0.06:
        p.append("too_dark")
    if m["colour"] > 0.03 or m["grey"] > 0.10:
        p.append("not_line_art")
    if m["ink"] < 0.012:
        p.append("too_empty")
    if m["border_ink"] > 0.35:
        p.append("frame")
    if v:
        if v.get("line_art", 10) < 6 and "not_line_art" not in p:
            p.append("not_line_art")
        if v.get("text"):
            p.append("text")
        if v.get("frame") and "frame" not in p:
            p.append("frame")
    return p


def verify(image, elements: List[str], use_vlm: bool = True, culture: str = "") -> dict:
    """VLM check + pixel metrics. Returns {"verified", "line_art", "text", "frame", "note",
    "metrics", "problems", "missing", "score"}."""
    m = metrics(image)
    elements = [e for e in elements if e]
    v = {"elements": {e: True for e in elements}, "line_art": 10, "text": False, "frame": False, "note": ""}
    if use_vlm and not config.mock("vlm"):
        from . import vlm
        from .art_glossary import verify_hint
        items = "\n".join("- %s%s" % (e, (": " + verify_hint(e, culture)) if verify_hint(e, culture) else "")
                          for e in elements)
        example = ", ".join('"%s": true' % e.replace('"', "'") for e in elements)
        try:
            d = vlm.ask_json([image], VERIFY_PROMPT.format(items=items, example=example),
                             schema={"elements": dict}, max_tokens=400)
            got = {str(k).strip().lower(): bool(val) for k, val in (d.get("elements") or {}).items()}
            v["elements"] = {e: got.get(e.lower(), _fuzzy(got, e)) for e in elements}
            v["line_art"] = _num(d.get("line_art"), 5)
            v["text"] = bool(d.get("text"))
            v["frame"] = bool(d.get("frame"))
            v["note"] = str(d.get("note") or "")[:200]
        except Exception as e:                           # a broken check must not stop the drawing
            log.warning("VLM verification failed: %s", e)
            v["note"] = "verification failed: %s" % type(e).__name__
            v["elements"] = {e: False for e in elements}
            v["unverified"] = True
    probs = problems_from(m, v)
    missing = [e for e, ok in v["elements"].items() if not ok]
    out = {"verified": v["elements"], "line_art": v["line_art"], "text": v["text"], "frame": v["frame"],
           "note": v["note"], "metrics": m, "problems": probs, "missing": missing}
    out["score"] = score(out)
    return out


def _num(x, default) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _fuzzy(got: dict, e: str) -> bool:
    e = e.lower()
    for k, val in got.items():
        if e in k or k in e:
            return val
    return False


def score(v: dict) -> float:
    els = v.get("verified") or {}
    frac = (sum(1 for ok in els.values() if ok) / float(len(els))) if els else 1.0
    s = 0.6 * frac + 0.3 * min(1.0, max(0.0, v.get("line_art", 5) / 10.0))
    probs = v.get("problems") or []
    s += 0.1 * (1.0 - min(1.0, 0.5 * len([p for p in probs if p in ("too_dark", "not_line_art", "too_empty")])))
    s -= 0.12 * ("text" in probs) + 0.08 * ("frame" in probs)
    return round(max(0.0, min(1.0, s)), 3)


def good_enough(v: dict) -> bool:
    return not v.get("missing") and not any(p in (v.get("problems") or []) for p in
                                            ("text", "not_line_art", "too_dark", "too_empty")) \
        and v.get("line_art", 0) >= 6
