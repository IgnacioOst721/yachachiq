"""Open LLM on the Mac (MLX). Owned by ART; other domains call chat() too.

    chat(messages, max_tokens=800, json_mode=False, temperature=0.2) -> str
    chat_json(messages, schema=None, ...) -> dict

messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
json_mode=True: the JSON requirement is added to the prompt, the reply is parsed
(fences, trailing commas and <think> blocks tolerated) and re-asked up to 2 times
quoting the error; the returned string is always valid JSON (else ValueError).

Model: YQ_ART_LLM (key below). Text models load with mlx-lm; a key that is also
in vlm.REPOS (a natively multimodal model) is served by vlm.chat_text so the LLM
and the VLM share ONE copy of the weights in memory.
Qwen3-family "thinking" is switched off (enable_thinking=False): planning quality
was the same in our tests and it is 3-5x faster (docs/art.md).
Mock: YQ_MOCK=1 or YQ_MOCK_LLM=1.
"""
from __future__ import annotations

import json
import logging
from typing import List

from yq.common import config
from yq.common.config import env
from yq.macworker.modelmgr import models

from . import vlm
from .art_runtime import extract_json, local_repo, run, strip_think, validate

log = logging.getLogger("yq.art.llm")

# key -> (hf repo, resident GB on the M4: weights + KV cache for ~4k tokens)
TEXT_REPOS = {
    "qwen3-8b": ("mlx-community/Qwen3-8B-4bit", 5.2),
    "qwen2.5-7b": ("mlx-community/Qwen2.5-7B-Instruct-4bit", 4.8),
}
LLM = env("ART_LLM", "qwen3-8b")


def _load(key: str):
    from mlx_lm import load
    model, tokenizer = load(local_repo(TEXT_REPOS[key][0]))
    return {"model": model, "tokenizer": tokenizer, "key": key}


def _unload(obj) -> None:
    obj.clear()


def register(key: str = None) -> str:
    key = key or LLM
    if key in TEXT_REPOS:
        name = "llm:" + key
        if name not in models.registered():
            models.register(name, loader=lambda: _load(key), size_gb=TEXT_REPOS[key][1], unloader=_unload,
                            repo=TEXT_REPOS[key][0], kind="llm", domain="art")
        return name
    if key in vlm.REPOS:
        return vlm.register(key)
    raise KeyError("unknown LLM %r (known: %s)" % (key, sorted(TEXT_REPOS) + sorted(vlm.REPOS)))


def _generate(key: str, messages: List[dict], max_tokens: int, temperature: float) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    m = models.get(register(key))
    tok = m["tokenizer"]
    try:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    sampler = make_sampler(temp=float(temperature), top_p=0.9 if temperature > 0 else 0.0)
    out = generate(m["model"], tok, prompt=prompt, max_tokens=int(max_tokens), sampler=sampler, verbose=False)
    return strip_think(out)


def _mock(messages: List[dict], json_mode: bool) -> str:
    last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if json_mode:
        return json.dumps({"mock": True, "echo": str(last)[:200]}, ensure_ascii=False)
    return "[mock llm] " + str(last)[:200]


def _raw(messages: List[dict], max_tokens: int, temperature: float, key: str) -> str:
    if key in TEXT_REPOS:
        return run(_generate, key, messages, max_tokens, temperature)
    return vlm.chat_text(messages, max_tokens=max_tokens, temperature=temperature, key=key)


_JSON_RULE = ("Reply with ONE valid JSON object only: no prose before or after, no code fences, "
              "double quotes for every key and string.")


def chat(messages: List[dict], max_tokens: int = 800, json_mode: bool = False, temperature: float = 0.2,
         schema: dict = None, key: str = None, retries: int = 2) -> str:
    """Chat completion. With json_mode the result is a JSON string (validated against the
    optional `schema` = {"key": python type})."""
    msgs = [dict(m) for m in messages]
    if config.mock("llm"):
        return _mock(msgs, json_mode)
    key = key or LLM
    if not json_mode:
        return _raw(msgs, max_tokens, temperature, key)
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n" + _JSON_RULE
    else:
        msgs.insert(0, {"role": "system", "content": _JSON_RULE})
    last = ""
    for attempt in range(retries + 1):
        text = _raw(msgs, max_tokens, temperature if attempt == 0 else max(0.1, temperature), key)
        last = text
        try:
            obj = extract_json(text)
            probs = validate(obj, schema) if schema else []
            if not probs:
                return json.dumps(obj, ensure_ascii=False)
            err = "; ".join(probs)
        except ValueError as e:
            err = str(e)[:200]
        log.info("LLM JSON retry %d: %s", attempt + 1, err)
        msgs = msgs + [{"role": "assistant", "content": text[:2000]},
                       {"role": "user", "content": "That reply was not valid (%s). Send the complete "
                                                   "JSON object again, fixed, and nothing else." % err}]
    log.warning("LLM JSON failed after %d tries (%s). Last reply: %r", retries + 1, err, last[-1500:])
    e = ValueError("LLM did not return valid JSON after %d tries (%s): %r" % (retries + 1, err, last[:300]))
    e.last_text = last                                   # callers may salvage fields from it
    raise e


def chat_json(messages: List[dict], schema: dict = None, max_tokens: int = 800, temperature: float = 0.2,
              key: str = None) -> dict:
    return json.loads(chat(messages, max_tokens=max_tokens, json_mode=True, temperature=temperature,
                           schema=schema, key=key))
