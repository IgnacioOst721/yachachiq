"""make_drawing: a visitor's story -> plotter files for both sides of the sheet (CONTRACTS.md §6).

    make_drawing(story, out_dir, on_progress=None) -> DrawingResult

With the MacBook: /story/plan (LLM) -> job "image" (generator + VLM check + retries) ->
vectorize here (CPU) -> front.svg/.gcode. Without it (or if anything on the Mac fails):
the v1 procedural motif composer, so a drawing ALWAYS comes out. The back page (title,
story text, Spanish translation, credit, QR) is made here in both cases.
Files in out_dir: image.png, front.svg, front.gcode, front.png, back.svg, back.gcode,
back.png, plan.json, drawing.json.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from yq.common import config
from yq.common.contracts import DrawingResult, Progress, ScenePlan, StoryInput, from_dict, to_dict

from . import gcode, layout, motifs, offline, order, qr, settings, svg, vectorize
from .geom import Stroke, length, total_length, travel

log = logging.getLogger("yq.art.drawing")


def _progress(cb: Optional[Callable], stage: str, fraction: float, msg: str, **detail) -> None:
    if cb:
        try:
            cb(Progress(stage=stage, fraction=round(float(fraction), 3), message_es=msg, detail=detail))
        except Exception:                                   # the UI must never break the drawing
            log.exception("progress callback failed")


def _from_mac(story: StoryInput, out: Path, cb) -> dict:
    """Plan + image on the Mac. Raises MacUnavailable/MacJobError/others on any failure."""
    from yq.common.macclient import MacUnavailable, client
    c = client()
    if not c.available():
        raise MacUnavailable("Mac worker not reachable")
    _progress(cb, "planning", 0.04, "Pensando en tu historia...")
    t0 = time.time()
    r = c.post_json("/story/plan", {"text": story.text, "lang": story.lang, "text_es": story.text_es,
                                    "text_en": story.text_en}, timeout=240.0)
    plan_s = time.time() - t0
    plan = from_dict(ScenePlan, r["plan"])
    _progress(cb, "generating", 0.12, "Imaginando el dibujo...", elements=plan.elements)
    jid = c.submit_job("image", {"plan": to_dict(plan), "width": settings.IMAGE_W, "height": settings.IMAGE_H,
                                 "max_attempts": settings.MAX_ATTEMPTS, "verify": True})
    try:
        res = c.wait_job(jid, on_progress=lambda f, m: _progress(cb, "generating", 0.12 + 0.6 * f,
                                                                 m or "Dibujando tu historia..."))
    except BaseException:
        try:                                  # do not leave the Mac busy with a picture nobody will use
            c._request("POST", "/jobs/%s/cancel" % jid, timeout=5.0)
        except Exception:
            pass
        raise
    c.download(jid, res.get("image", "image.png"), out / "image.png")
    secs = dict(res.get("seconds") or {})
    secs.update({"plan": round(plan_s, 1), "image_job": round(time.time() - t0 - plan_s, 1)})
    return {"plan": plan, "text_es": r.get("text_es") or "", "text_en": r.get("text_en") or "", "seconds": secs,
            "verified": res.get("verified") or {}, "attempts": int(res.get("attempts") or 1),
            "score": res.get("score"), "backend": res.get("backend", ""), "job": jid}


def _limit_time(strokes: List[Stroke], start, minutes: float) -> List[Stroke]:
    """If the plotter would take too long, drop the shortest strokes (details go last)."""
    est = gcode.estimate(strokes, start)["seconds"] / 60.0
    if est <= minutes or len(strokes) < 20:
        return strokes
    lens = np.array([length(s) for s in strokes])
    for q in (5, 10, 15, 20, 30, 40):
        thr = np.percentile(lens, q)
        kept = [s for s, L in zip(strokes, lens) if L > thr]
        if gcode.estimate(kept, start)["seconds"] / 60.0 <= minutes:
            return kept
    return kept


def _finish(strokes: List[Stroke], start) -> tuple:
    naive = travel(strokes, start)
    opt = order.optimize(strokes, start, budget_s=3.0)
    info = {"travel_naive_mm": round(naive, 1), "travel_mm": round(travel(opt, start), 1)}
    if settings.continuous():
        route, visible = order.continuous(opt, start)
        info["visible_hops_mm"] = round(visible, 1)
        return route, info
    return opt, info


def make_drawing(story: StoryInput, out_dir, on_progress: Optional[Callable] = None,
                 published: bool = True, use_mac: Optional[bool] = None) -> DrawingResult:
    """published=False (the visitor declined the web gallery): the QR points to the gallery home."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    times: dict = {}
    text_es = story.text_es or (story.text if story.lang.startswith("spa") else "")
    mac = None
    if use_mac is None:
        use_mac = not config.mock("mac")
    if use_mac:
        try:
            mac = _from_mac(story, out, on_progress)
        except Exception as e:
            log.warning("Mac drawing unavailable (%s: %s): using the offline motifs", type(e).__name__, e)
            _progress(on_progress, "offline", 0.1, "Dibujando sin la computadora grande...")
    times["mac_s"] = round(time.time() - t0, 1)
    if mac:
        times.update({k + "_s": v for k, v in (mac.get("seconds") or {}).items()})
    origin = gcode.origin_page()
    t1 = time.time()
    if mac:
        plan = mac["plan"]
        text_es = text_es or mac["text_es"]
        _progress(on_progress, "tracing", 0.75, "Convirtiendo el dibujo en trazos de lápiz...")
        tr = vectorize.trace(out / "image.png")
        front, finfo = layout.front_page(tr.strokes, tr.width, tr.height, plan.title)
        finfo.update(tr.info)
        source = "mac:" + mac.get("backend", "")
        if len(front) < 8 or total_length(front) < 150.0:      # blank or unusable picture
            log.warning("generated picture traced to almost nothing: offline motifs instead")
            mac = None
    if not mac:
        plan = offline.plan(story.text, story.lang, story.text_es, story.text_en)
        els = offline.find_elements(" ".join([story.text, story.text_es, story.text_en])) or ["mountain", "sun", "llama"]
        seed = sum(ord(c) for c in story.story_id) & 0xFFFF
        x0, y0, x1, y1 = layout.front_box(False)
        hh = min(y1 - y0, (x1 - x0) * 0.95)                   # v1 composer was made for wide pages
        top = y0 + ((y1 - y0) - hh) / 2.0
        front = motifs.compose_page(els[:6], (x0, top, x1, top + hh), seed=seed)
        finfo = {"motifs": els}
        svg.preview_png({"black": front}, out / "image.png")
        source = "offline"
    front = _limit_time(front, origin, settings.MAX_DRAW_MINUTES)
    front, fo = _finish(front, origin)
    finfo.update(fo)
    times["front_s"] = round(time.time() - t1, 1)
    _progress(on_progress, "layout", 0.88, "Escribiendo tu historia en la parte de atrás...")
    t2 = time.time()
    url = qr.story_url(story.story_id, published)
    back_raw, binfo = layout.back_page(story.text, story.lang, text_es, plan.title or plan.title_es, url)
    back, bo = _finish(back_raw, origin)
    binfo.update(bo)
    times["back_s"] = round(time.time() - t2, 1)
    pens = settings.PENS[0] if settings.PENS else "black"
    files = {}
    for side, strokes in (("front", front), ("back", back)):
        (out / (side + ".svg")).write_text(svg.to_svg({pens: strokes}, title="%s - %s" % (story.story_id, side)))
        gcode.save(gcode.to_gcode(strokes, "%s %s" % (story.story_id, side)), out / (side + ".gcode"))
        svg.preview_png({pens: strokes}, out / (side + ".png"))
        files[side] = gcode.estimate(strokes, origin)
    est_min = (files["front"]["seconds"] + files["back"]["seconds"]) / 60.0
    (out / "plan.json").write_text(json.dumps(to_dict(plan), ensure_ascii=False, indent=1))
    verified = (mac or {}).get("verified") or {}
    report = {"story_id": story.story_id, "title": plan.title, "title_es": plan.title_es, "source": source,
              "qr_url": url, "published": bool(published), "times": times,
              "front": {**finfo, **files["front"], "strokes": len(front)},
              "back": {**binfo, **files["back"], "strokes": len(back)},
              "verified": verified, "attempts": (mac or {}).get("attempts", 0), "score": (mac or {}).get("score"),
              "total_s": round(time.time() - t0, 1)}
    (out / "drawing.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    _progress(on_progress, "done", 1.0, "¡Tu dibujo está listo!")
    return DrawingResult(story_id=story.story_id, image=str(out / "image.png"), front_svg=str(out / "front.svg"),
                         back_svg=str(out / "back.svg"), front_gcode=str(out / "front.gcode"),
                         back_gcode=str(out / "back.gcode"), verified=verified,
                         attempts=int((mac or {}).get("attempts", 1) or 1), strokes=len(front),
                         pen_mm=round(total_length(front) + total_length(back), 1),
                         est_minutes=round(est_min, 1), qr_url=url)
