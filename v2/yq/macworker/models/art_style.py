"""Prompt style per image generator: what makes each model draw clean black ink line art
on pure white that vectorizes well (tuned by the runs in tools/art_eval.py, docs/art.md).

    compose_prompt(scene, backend)                    -> final positive prompt
    emphasize(scene, missing, problems, attempt)      -> scene rewritten for a retry
    negative(backend)                                 -> negative prompt (only SD1.5 uses one)
"""
from __future__ import annotations

from yq.common.config import env

BACKEND = env("ART_IMAGE_BACKEND", "flux2-klein-4b")

# Distilled flow models (Z-Image-Turbo, FLUX.2-klein, FLUX.1-schnell) ignore negative prompts, so
# everything is said positively. Short, concrete style words worked better than long lists.
_FLOW_STYLE = ("Monochrome black ink line art on pure white paper, like a page of a children's colouring "
               "book drawn with a fine black pen: only black outlines, every shape left white inside, "
               "clean confident lines of even thickness, clear silhouettes, no shading, no grey, no colour "
               "at all, no solid black areas, no text, no border. The whole scene fits inside the picture "
               "with white margins.")

_SD15_STYLE = ("(coloring book style:1.3), (black outline drawing on white paper:1.3), (white background:1.4), "
               "no frame, no border, clean thin outlines only, no shading, no fill, no color, centered, one scene")
_SD15_NEGATIVE = ("(black background:1.6), dark background, inverted colors, vignette, circle frame, "
                  "(photograph:1.4), (photorealistic:1.4), grayscale photo, 3d render, color, colored, painting, "
                  "shading, gradient, filled areas, solid black, hatching, crosshatch, texture, pattern, "
                  "decorative border, frame, text, watermark, signature, letters, blurry, noise, cluttered")


def is_sd15(backend: str = None) -> bool:
    return (backend or BACKEND) in ("comfyui", "sd15")


def compose_prompt(scene: str, backend: str = None) -> str:
    scene = " ".join((scene or "").split()).rstrip(".")
    if is_sd15(backend):
        # SD1.5 weighs the first tokens most: subject first, then the style (v1 lesson)
        return "a simple line drawing of %s, %s" % (scene, _SD15_STYLE)
    return "%s. %s" % (scene, _FLOW_STYLE)


def negative(backend: str = None) -> str:
    return _SD15_NEGATIVE if is_sd15(backend) else ""


def scene_of(prompt: str, backend: str = None) -> str:
    """Recover the scene part of a composed prompt."""
    p = prompt or ""
    if is_sd15(backend):
        p = p.replace("a simple line drawing of ", "", 1).split(", " + _SD15_STYLE[:20])[0]
        return p
    return p.split(". " + _FLOW_STYLE[:30])[0]


def emphasize(scene: str, missing: list, problems: list, attempt: int, hints: dict = None) -> str:
    """Retry prompt: missing elements first and explicit, plus fixes for the style problems."""
    s = " ".join((scene or "").split()).rstrip(".")
    if missing:
        shown = ["%s (%s)" % (m, hints[m]) if hints and hints.get(m) else m for m in missing]
        s = "Clearly showing %s. %s" % (" and ".join(shown), s)
    fixes = []
    if "not_line_art" in problems or "too_dark" in problems:
        fixes.append("only thin black outlines on white, large empty white areas")
    if "text" in problems:
        fixes.append("absolutely no letters or writing anywhere")
    if "frame" in problems:
        fixes.append("no frame and no border around the picture")
    if "too_empty" in problems:
        fixes.append("the subject is large and fills most of the picture")
    if fixes:
        s += ". " + ", ".join(fixes)
    return s
