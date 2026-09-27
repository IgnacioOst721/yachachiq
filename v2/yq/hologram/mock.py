"""Hologram simulator shared by the mock HTTP server and the in-process mock client."""
from __future__ import annotations

import threading
import time
from typing import Optional

from yq.hologram.client import HologramError


class HologramSim:
    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self.stories: dict = {}
        self.playing: Optional[str] = None
        self._until = 0.0
        self._lock = threading.Lock()

    def add(self, story: dict, files: dict) -> dict:
        for s in story.get("scenes", []):
            for key in ("image", "audio"):
                if s.get(key) and s[key] not in files:
                    raise HologramError("missing %s" % s[key])
        dur = 6.0 + 0.04 * len(story.get("text", ""))
        with self._lock:
            self.stories[story["story_id"]] = {"story": story, "files": {k: len(v) for k, v in files.items()}}
            self.playing = story["story_id"]
            self._until = time.time() + dur * self.speed
        return {"ok": True, "id": story["story_id"], "duration_s": round(dur * self.speed, 1)}

    def status(self) -> dict:
        with self._lock:
            left = self._until - time.time()
            if left <= 0:
                self.playing = None
            return {"playing": self.playing, "remaining_s": round(max(0.0, left), 1)}


class MockHologramClient:
    source = "mock:ui"

    def __init__(self, speed: Optional[float] = None):
        from yq.server import settings
        self.sim = HologramSim(speed if speed is not None else float(settings.MOCK_SPEED))
        self.url = "mock://hologram"
        self.online = True

    def available(self, max_age_s: float = 5.0) -> bool:
        return self.online

    def send(self, story: dict, files: dict) -> dict:
        if not self.online:
            raise HologramError("holograma no disponible (mock apagado)")
        return self.sim.add(story, files)

    def status(self) -> dict:
        return self.sim.status()
