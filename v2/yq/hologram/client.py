"""Client for Joaquín's hologram (Looking Glass Go + speakers), CONTRACTS.md §8 (proposal).

    POST {HOLOGRAM_URL}/stories   multipart, one part per file, field name = file name:
         story.json = {"story_id","title","lang","text","text_es",
                       "scenes":[{"image":"scene_1.png","caption_es":"...","audio":"narration_1.wav"}]}
         + every image / audio file referenced by the scenes
         -> {"ok": true, "id": "<story id>", "duration_s"?: float}   (the hologram plays it right away)
    GET  {HOLOGRAM_URL}/status     -> {"playing": story_id|null, "remaining_s"?: float}   (optional)
    GET  {HOLOGRAM_URL}/health     (optional; any HTTP answer = reachable)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from yq.common import config


class HologramError(RuntimeError):
    pass


def build_story(story_id: str, title: str, lang: str, text: str, text_es: str, scenes: list) -> dict:
    """scenes: [{"image": path, "caption_es": str, "audio": path|""}] -> (story.json dict, files dict)."""
    return {"story_id": story_id, "title": title, "lang": lang, "text": text, "text_es": text_es,
            "scenes": [{"image": Path(s["image"]).name if s.get("image") else "",
                        "caption_es": s.get("caption_es", ""),
                        "audio": Path(s["audio"]).name if s.get("audio") else ""} for s in scenes]}


def package_files(scenes: list) -> dict:
    files = {}
    for s in scenes:
        for key in ("image", "audio"):
            p = s.get(key)
            if p and Path(p).exists():
                files[Path(p).name] = Path(p).read_bytes()
    return files


class HologramClient:
    source = "http"

    def __init__(self, url: Optional[str] = None, timeout: Optional[float] = None):
        self.url = (url or config.HOLOGRAM_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else config.HTTP_TIMEOUT
        self._probe = (0.0, False)

    def _req(self, method, path, timeout=None, **kw):
        import requests
        try:
            return requests.request(method, self.url + path, timeout=timeout or self.timeout, **kw)
        except requests.RequestException as e:
            raise HologramError("holograma no disponible en %s (%s)" % (self.url, e.__class__.__name__))

    def available(self, max_age_s: float = 5.0) -> bool:
        t, ok = self._probe
        if time.time() - t < max_age_s:
            return ok
        try:
            ok = self._req("GET", "/health", timeout=2.0).status_code < 500
        except HologramError:
            ok = False
        self._probe = (time.time(), ok)
        return ok

    def send(self, story: dict, files: dict) -> dict:
        """story: story.json dict; files: {name: bytes}. Returns the hologram's answer."""
        missing = [n for s in story.get("scenes", []) for n in (s.get("image"), s.get("audio")) if n and n not in files]
        if missing:
            raise HologramError("faltan archivos del paquete: %s" % ", ".join(missing))
        parts = [("story.json", ("story.json", json.dumps(story, ensure_ascii=False).encode("utf-8"),
                                 "application/json"))]
        for name, data in files.items():
            ctype = "audio/wav" if name.lower().endswith(".wav") else "image/png" if name.lower().endswith(".png") \
                else "image/jpeg"
            parts.append((name, (name, data, ctype)))
        r = self._req("POST", "/stories", files=parts, timeout=max(self.timeout, 60.0))
        if r.status_code >= 400:
            raise HologramError("el holograma rechazó la historia (HTTP %d): %s" % (r.status_code, r.text[:200]))
        try:
            return r.json()
        except ValueError:
            return {"ok": True}

    def status(self) -> dict:
        try:
            r = self._req("GET", "/status", timeout=3.0)
            return r.json() if r.status_code < 400 else {}
        except (HologramError, ValueError):
            return {}
