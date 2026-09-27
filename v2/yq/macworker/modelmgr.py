"""Loads AI models on demand on the MacBook and evicts the least recently used
ones when the memory budget (config.MAC_MODEL_BUDGET_GB) would be exceeded.

Each domain registers its models once at import time:

    from yq.macworker.modelmgr import models
    models.register("whisper-large-v3", loader=_load_whisper, size_gb=3.2)
    ...
    whisper = models.get("whisper-large-v3")      # loads if needed

`loader()` returns any object; an optional `unloader(obj)` frees it. After an
eviction the MLX / PyTorch caches are emptied so the memory really returns.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from yq.common import config

log = logging.getLogger("yq.models")


@dataclass
class _Entry:
    name: str
    loader: Callable[[], Any]
    size_gb: float
    unloader: Optional[Callable[[Any], None]] = None
    obj: Any = None
    last_used: float = 0.0
    load_seconds: float = 0.0
    meta: dict = field(default_factory=dict)


def free_accelerator_caches() -> None:
    gc.collect()
    try:
        import mlx.core as mx
        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear:
            clear()
    except Exception:
        pass
    try:
        import torch
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class ModelManager:
    def __init__(self, budget_gb: float = None):
        self.budget_gb = budget_gb if budget_gb is not None else config.MAC_MODEL_BUDGET_GB
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.RLock()

    def register(self, name: str, loader: Callable[[], Any], size_gb: float,
                 unloader: Optional[Callable[[Any], None]] = None, **meta) -> None:
        with self._lock:
            old = self._entries.get(name)
            if old and old.obj is not None:
                return  # already registered and loaded: keep it
            self._entries[name] = _Entry(name, loader, float(size_gb), unloader, meta=meta)

    def registered(self) -> list[str]:
        return sorted(self._entries)

    def loaded(self) -> list[str]:
        with self._lock:
            return [e.name for e in self._entries.values() if e.obj is not None]

    def used_gb(self) -> float:
        with self._lock:
            return sum(e.size_gb for e in self._entries.values() if e.obj is not None)

    def get(self, name: str) -> Any:
        with self._lock:
            if name not in self._entries:
                raise KeyError("model %r not registered (known: %s)" % (name, self.registered()))
            e = self._entries[name]
            if e.obj is None:
                self._make_room(e.size_gb, keep=name)
                t0 = time.time()
                log.info("loading %s (~%.1f GB)", name, e.size_gb)
                e.obj = e.loader()
                e.load_seconds = time.time() - t0
                log.info("loaded %s in %.1f s", name, e.load_seconds)
            e.last_used = time.time()
            return e.obj

    def unload(self, name: str) -> None:
        with self._lock:
            e = self._entries.get(name)
            if not e or e.obj is None:
                return
            log.info("unloading %s", name)
            try:
                if e.unloader:
                    e.unloader(e.obj)
            finally:
                e.obj = None
                free_accelerator_caches()

    def unload_all(self) -> None:
        for n in self.loaded():
            self.unload(n)

    def _make_room(self, need_gb: float, keep: str) -> None:
        while self.used_gb() + need_gb > self.budget_gb:
            victims = sorted((e for e in self._entries.values() if e.obj is not None and e.name != keep),
                             key=lambda e: e.last_used)
            if not victims:
                break  # a single model bigger than the budget: load anyway, macOS will swap
            self.unload(victims[0].name)

    def stats(self) -> dict:
        with self._lock:
            return {
                "budget_gb": self.budget_gb,
                "used_gb": round(self.used_gb(), 2),
                "models": {e.name: {"loaded": e.obj is not None, "size_gb": e.size_gb,
                                    "load_seconds": round(e.load_seconds, 1), **e.meta}
                           for e in self._entries.values()},
            }


models = ModelManager()
