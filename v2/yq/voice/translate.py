"""Translation on the Jetson = ask the Mac (CONTRACTS.md §6).

    from yq.voice.translate import translate
    translate("Ñuqaqa runam kani", "quy_Latn", "spa_Latn")   -> "Yo soy una persona"

Falls back to returning the text unchanged when the Mac is unavailable (the
kiosk still shows the original story; nothing is invented).
"""
from __future__ import annotations

import logging

from yq.common import config, languages

from . import settings

log = logging.getLogger("yq.voice.translate")

last_engine = ""


def translate_detail(text: str, src: str, tgt: str) -> dict:
    """{"text", "engine"}; engine "same" / "none" when nothing was translated."""
    global last_engine
    text = (text or "").strip()
    s = languages.resolve(src) or src
    t = languages.resolve(tgt) or tgt
    if not text or s == t:
        last_engine = "same"
        return {"text": text, "engine": "same"}
    if config.mock("mac"):
        last_engine = "none"
        return {"text": text, "engine": "none"}
    try:
        from yq.common.macclient import client
        d = client().post_json("/translate", {"text": text, "src": s, "tgt": t}, timeout=settings.TRANSLATE_TIMEOUT_S)
        last_engine = d.get("engine", "")
        return {"text": d.get("text", text), "engine": last_engine}
    except Exception as e:
        log.warning("translate %s->%s failed: %s", s, t, e)
        last_engine = "none"
        return {"text": text, "engine": "none"}


def translate(text: str, src: str, tgt: str) -> str:
    return translate_detail(text, src, tgt)["text"]
