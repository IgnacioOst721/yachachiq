"""Shared plumbing for the ART models on the Mac (LLM, VLM, image generator).

    run(fn, *args)          run on the single "mlx" thread (MLX streams are per thread, and
                            one GPU user at a time keeps the 16 GB Mac out of swap)
    local_repo(repo_id)     path of an already-downloaded Hugging Face repo (never touches the
                            network unless YQ_ART_ALLOW_DOWNLOAD=1)
    extract_json(text)      first JSON object/array in a model reply (fences, trailing commas...)
    strip_think(text)       remove <think>...</think> blocks (Qwen3 family)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from yq.common.config import env
from yq.macworker.mlx_thread import run  # noqa: F401  (the worker-wide MLX thread, shared with voice)


def local_repo(repo_id: str) -> str:
    """Local snapshot folder of repo_id. Runtime is offline: we only read the HF cache."""
    if os.path.isdir(repo_id):
        return repo_id
    from huggingface_hub import snapshot_download
    if env("ART_ALLOW_DOWNLOAD", False):
        return snapshot_download(repo_id)
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception as e:
        raise RuntimeError("model %s is not downloaded; run: .venvs/art/bin/hf download %s  (%s)"
                           % (repo_id, repo_id, type(e).__name__))


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "<think>" in text:                     # unterminated thinking (ran out of tokens)
        text = text.split("<think>")[0]
    return text.strip()


def _balanced(s: str, start: int) -> str:
    open_c = s[start]
    close_c = "}" if open_c == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s[start:]


def _repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)                       # trailing commas
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    depth_curly = s.count("{") - s.count("}")
    depth_square = s.count("[") - s.count("]")
    if depth_square > 0:
        s += "]" * depth_square
    if depth_curly > 0:
        s += "}" * depth_curly
    return s


def extract_json(text: str) -> Any:
    """Parse the first JSON object (or array) in `text`. Raises ValueError when there is none."""
    t = strip_think(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(t)
    last_err = None
    for c in candidates:
        for m in re.finditer(r"[\{\[]", c):
            chunk = _balanced(c, m.start())
            for attempt in (chunk, _repair(chunk)):
                try:
                    return json.loads(attempt)
                except ValueError as e:
                    last_err = e
            break                                # only the first opening bracket of each candidate
    raise ValueError("no JSON found in model reply (%s): %r" % (last_err, (text or "")[:200]))


def validate(obj: Any, schema: dict) -> list:
    """Tiny schema check: {"key": type or (types,)} -> list of problems (empty = ok)."""
    problems = []
    if not isinstance(obj, dict):
        return ["the reply must be a JSON object"]
    for key, typ in (schema or {}).items():
        if key not in obj:
            problems.append("missing key %r" % key)
        elif typ is not None and not isinstance(obj[key], typ):
            problems.append("key %r must be %s" % (key, getattr(typ, "__name__", typ)))
    return problems
