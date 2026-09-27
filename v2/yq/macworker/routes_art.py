"""ART endpoints on the Mac worker (CONTRACTS.md §3.2).

    POST /story/clean  {"text","lang"}                          -> {"text"}
    POST /story/plan   {"text","lang","text_es"?,"text_en"?}    -> {"plan": ScenePlan, "text_es", "text_en"}
    job "image"        params {"plan", "width", "height", "seed"?, "max_attempts"?, "verify": true, "backend"?}
                       -> {"image": "image.png", "attempts", "verified": {element: bool}, "score", ...}

The image job loops generate -> verify (VLM + pixel metrics) -> regenerate with a new seed
and a prompt that stresses what was missing, keeps the best attempt by score, and stops
early when everything is there and the style is clean.
"""
from __future__ import annotations

import json
import logging
import random
import shutil
import time

from fastapi import APIRouter, Body, HTTPException

from yq.common.config import env
from yq.common.contracts import ScenePlan, from_dict, to_dict

log = logging.getLogger("yq.art.routes")
router = APIRouter()

FALLBACK_BACKENDS = env("ART_IMAGE_FALLBACKS", ["comfyui"])
# attempt 1 uses the fast generator; if the check fails, retries use this more accurate one ("" = off)
ESCALATE_BACKEND = env("ART_IMAGE_ESCALATE", "z-image-turbo")


@router.post("/story/clean")
def story_clean(payload: dict = Body(...)):
    from yq.macworker.models import art_plan
    text = str(payload.get("text") or "")
    try:
        return {"text": art_plan.clean(text, str(payload.get("lang") or "spa_Latn"))}
    except Exception as e:
        log.exception("clean failed")
        return {"text": text, "error": "%s: %s" % (type(e).__name__, e)}


@router.post("/story/plan")
def story_plan(payload: dict = Body(...)):
    from yq.macworker.models import art_plan
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is empty")
    t0 = time.time()
    try:
        p, es, en = art_plan.plan(text, str(payload.get("lang") or "spa_Latn"), str(payload.get("text_es") or ""),
                                  str(payload.get("text_en") or ""), backend=payload.get("backend"))
    except Exception as e:
        log.exception("plan failed")
        raise HTTPException(500, "plan failed: %s: %s" % (type(e).__name__, e))
    return {"plan": to_dict(p), "text_es": es, "text_en": en, "seconds": round(time.time() - t0, 2)}


def _generate_any(prompt: str, negative: str, w: int, h: int, seed: int, backend: str, progress=None,
                  cancelled=None):
    from yq.macworker.models import art_image, art_style
    order = [backend] + [b for b in FALLBACK_BACKENDS if b != backend]
    last = None
    for b in order:
        try:
            p = prompt if b == backend else art_style.compose_prompt(art_style.scene_of(prompt, backend), b)
            kw = {"progress": progress, "cancelled": cancelled} if b != "comfyui" else {}
            return art_image.generate(p, w, h, seed, backend=b, negative=negative or art_style.negative(b), **kw), b
        except art_image.Cancelled:
            raise
        except Exception as e:
            last = e
            log.warning("image backend %s failed: %s", b, e)
    raise RuntimeError("every image backend failed: %s" % last)


def image_job(ctx) -> dict:
    from yq.macworker.models import art_image, art_style, art_verify
    prm = ctx.params
    plan = from_dict(ScenePlan, prm.get("plan") or {})
    backend = str(prm.get("backend") or art_image.BACKEND)
    w, h = int(prm.get("width") or 768), int(prm.get("height") or 1088)
    max_attempts = max(1, int(prm.get("max_attempts") or 3))
    do_verify = bool(prm.get("verify", True))
    seed0 = int(prm["seed"]) if prm.get("seed") is not None else random.randint(0, 2 ** 31 - 1000)
    scene = art_style.scene_of(plan.prompt, backend) or plan.subject
    culture = next((n.split(":", 1)[1].strip() for n in plan.cultural_notes or []
                    if str(n).lower().startswith("culture:")), "")
    from yq.macworker.models.art_glossary import verify_hint
    hints = {e: verify_hint(e, culture) for e in plan.elements}
    prompt = plan.prompt or art_style.compose_prompt(scene, backend)
    tried, best, used, last_v = [], None, backend, {}
    for k in range(max_attempts):
        if ctx.cancelled:
            break
        base = 0.05 + 0.9 * k / max_attempts
        step = 0.9 / max_attempts
        ctx.progress(base, "Dibujando tu historia..." if k == 0 else "Mejorando el dibujo (intento %d)..." % (k + 1))
        if k > 0:
            # escalate: the fast generator failed the check, so retries use the more accurate one
            # (FLUX.2 klein draws a condor with an eagle's head; Z-Image draws the bald head and ruff)
            if ESCALATE_BACKEND and ESCALATE_BACKEND != backend and not prm.get("backend"):
                backend = ESCALATE_BACKEND
            prompt = art_style.compose_prompt(art_style.emphasize(scene, last_v.get("missing", []),
                                                                  last_v.get("problems", []), k, hints), backend)
        t0 = time.time()
        img, used = _generate_any(prompt, plan.negative, w, h, seed0 + 1000 * k, backend,
                                  progress=lambda f, b=base, s=step: ctx.progress(b + s * 0.7 * f),
                                  cancelled=lambda: ctx.cancelled)
        t_gen = time.time() - t0
        name = "attempt_%d.png" % (k + 1)
        img.save(ctx.out_dir / name)
        ctx.progress(base + step * 0.7, "Revisando que el dibujo cuente tu historia...")
        t0 = time.time()
        v = art_verify.verify(img, plan.elements, use_vlm=do_verify, culture=culture)
        rec = {"file": name, "seed": seed0 + 1000 * k, "backend": used, "prompt": prompt,
               "gen_s": round(t_gen, 1), "verify_s": round(time.time() - t0, 1), **v}
        tried.append(rec)
        if best is None or v["score"] > best["score"]:
            best = rec
        last_v = v
        if art_verify.good_enough(v):
            break
    if best is None:
        raise RuntimeError("cancelled before any image was made")
    shutil.copyfile(ctx.out_dir / best["file"], ctx.out_dir / "image.png")
    (ctx.out_dir / "verify.json").write_text(json.dumps(tried, ensure_ascii=False, indent=1))
    ctx.progress(1.0, "Dibujo listo")
    return {"image": "image.png", "attempts": len(tried), "verified": best["verified"], "score": best["score"],
            "backend": best["backend"], "seed": best["seed"], "prompt": best["prompt"],
            "problems": best["problems"], "chosen": best["file"],
            "seconds": {"generate": round(sum(t["gen_s"] for t in tried), 1),
                        "verify": round(sum(t["verify_s"] for t in tried), 1)}}


def setup(jobs, models) -> None:
    from yq.macworker.models import art_image, llm, vlm
    jobs.register("image", image_job, heavy=True)
    regs = [llm.register, vlm.register]
    if art_image.BACKEND in art_image.BACKENDS and art_image.BACKEND != "comfyui":
        regs.append(lambda: art_image.register(art_image.BACKEND))
    for fn in regs:
        try:                                   # registration only records the loader: nothing loads here
            fn()
        except Exception as e:
            log.warning("model registration: %s", e)
