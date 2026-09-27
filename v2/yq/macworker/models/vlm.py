"""Vision-language model on the Mac (MLX, mlx-vlm). Owned by ART, used by BOX-ANALYSIS too.

    ask(images, prompt, max_tokens=600) -> str        images: paths or PIL images (0..n)
    ask_json(images, prompt, schema=None) -> dict     JSON reply, parsed/validated with one retry
    chat_text(messages, ...) -> str                   text-only chat on the same model (llm.py uses it
                                                      when the chosen LLM is this multimodal model)

Model: settings key YQ_ART_VLM (default below, chosen by measurement, see docs/art.md).
Images are converted to RGB and downscaled so the longest side is YQ_ART_VLM_MAX_SIDE
(default 896 px): Qwen-VL token count grows with pixels, and 896 keeps a 4-image call
under ~20 s on the M4 while objects stay recognisable.
Mock: YQ_MOCK=1 or YQ_MOCK_VLM=1 -> deterministic canned replies, no model loaded.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from yq.common import config
from yq.common.config import env
from yq.macworker.modelmgr import models

from .art_runtime import extract_json, local_repo, run, strip_think, validate

log = logging.getLogger("yq.art.vlm")

# key -> (hf repo, resident size in GB measured on the M4 incl. vision tower + cache headroom)
REPOS = {
    "qwen3.5-9b": ("mlx-community/Qwen3.5-9B-MLX-4bit", 6.8),
    "qwen3-vl-8b": ("mlx-community/Qwen3-VL-8B-Instruct-4bit", 6.6),
    "qwen2.5-vl-7b": ("mlx-community/Qwen2.5-VL-7B-Instruct-4bit", 6.4),
}
VLM = env("ART_VLM", "qwen3-vl-8b")
MAX_SIDE = env("ART_VLM_MAX_SIDE", 896)


def model_name(key: str = None) -> str:
    return "mm:" + (key or VLM)


def _load(key: str):
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    from pathlib import Path
    path = local_repo(REPOS[key][0])
    model, processor = load(path)
    cfg = load_config(Path(path))
    return {"model": model, "processor": processor, "config": cfg, "key": key}


def _unload(obj) -> None:
    obj.clear()


def register(key: str = None) -> str:
    key = key or VLM
    if key not in REPOS:
        raise KeyError("unknown VLM %r (known: %s)" % (key, sorted(REPOS)))
    name = model_name(key)
    if name not in models.registered():
        models.register(name, loader=lambda: _load(key), size_gb=REPOS[key][1], unloader=_unload,
                        repo=REPOS[key][0], kind="vlm", domain="art")
    return name


def get(key: str = None) -> dict:
    return models.get(register(key))


def _prepare(images: Optional[list], max_side: int) -> list:
    from PIL import Image
    out = []
    for im in images or []:
        if not hasattr(im, "convert"):
            im = Image.open(str(im))
        im = im.convert("RGB")
        w, h = im.size
        s = max_side / float(max(w, h))
        if s < 1.0:
            im = im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.LANCZOS)
        out.append(im)
    return out


def _generate(key: str, prompt: Any, images: list, max_tokens: int, temperature: float) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    m = get(key)
    formatted = apply_chat_template(m["processor"], m["config"], prompt, num_images=len(images),
                                    enable_thinking=False)
    res = generate(m["model"], m["processor"], formatted, images or None, max_tokens=max_tokens,
                   temperature=temperature, verbose=False)
    text = getattr(res, "text", res)
    return strip_think(text if isinstance(text, str) else str(text))


def _mock_reply(prompt: str, n_images: int) -> str:
    if "json" in (prompt or "").lower():
        return '{"mock": true, "images": %d}' % n_images
    return "[mock vlm] %d image(s): %s" % (n_images, (prompt or "")[:120])


def ask(images: list, prompt: str, max_tokens: int = 600, temperature: float = 0.0, key: str = None,
        max_side: int = None) -> str:
    """Answer `prompt` about the images (paths or PIL images, any size, 0..n of them)."""
    imgs_in = list(images or [])
    if config.mock("vlm"):
        return _mock_reply(prompt, len(imgs_in))
    key = key or VLM
    imgs = _prepare(imgs_in, int(max_side or MAX_SIDE))
    return run(_generate, key, prompt, imgs, max_tokens, temperature)   # the MLX thread serialises calls


def ask_json(images: list, prompt: str, schema: dict = None, max_tokens: int = 600, retries: int = 1,
             key: str = None) -> dict:
    """ask() that must return a JSON object: the prompt asks for JSON, the reply is parsed
    (code fences, trailing commas tolerated) and checked against `schema`
    ({"key": type}); on failure it asks once more quoting the problem."""
    p = prompt.rstrip() + "\n\nReply with ONE valid JSON object only (no prose, no code fences)."
    last = ""
    for attempt in range(retries + 1):
        text = ask(images, p, max_tokens=max_tokens, key=key)
        last = text
        try:
            obj = extract_json(text)
            probs = validate(obj, schema) if schema else []
            if not probs:
                return obj
            err = "; ".join(probs)
        except ValueError as e:
            err = str(e)[:200]
        p = (prompt.rstrip() + "\n\nYour previous reply was invalid (%s). Reply with ONE valid JSON "
             "object only." % err)
    raise ValueError("VLM did not return valid JSON: %r" % last[:300])


def chat_text(messages: List[dict], max_tokens: int = 800, temperature: float = 0.2, key: str = None) -> str:
    """Text-only chat through the multimodal model (system/user/assistant messages)."""
    key = key or VLM
    return run(_generate, key, [dict(m) for m in messages], [], max_tokens, temperature)
