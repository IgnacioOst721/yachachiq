"""Event bus: worker threads emit typed events, every WebSocket client receives them.

    bus.emit("progress", flow="story", stage="generating", fraction=0.4)

Events are plain dicts {"type": ..., "t": epoch, ...} (see docs/ui.md for the list).
emit() is safe from any thread. Each connected WebSocket owns an asyncio.Queue
fed with loop.call_soon_threadsafe, so a slow client never blocks a flow. The
last few hundred events are kept in `history` (tests read it; the gear menu can
show recent errors).
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("yq.server.bus")

# High-rate events: only the newest one matters, so a slow client drops the older ones.
COALESCE = {"level", "sign", "consent_tick", "scan_live", "printer"}
QUEUE_MAX = 400


class Subscriber:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)

    def push(self, msg: dict) -> None:
        def put():
            q = self.queue
            if q.full():
                try:                      # drop the oldest instead of blocking the producer
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(msg)
        try:
            self.loop.call_soon_threadsafe(put)
        except RuntimeError:              # loop closed (client gone / shutdown)
            pass


class EventBus:
    def __init__(self, history: int = 500):
        self._lock = threading.Lock()
        self._subs: set = set()
        self._listeners: list = []
        self.history: collections.deque = collections.deque(maxlen=history)
        self._last_emit: dict = {}

    # -- producers ---------------------------------------------------------------
    def emit(self, type_: str, min_interval: float = 0.0, **data) -> Optional[dict]:
        """Send an event to every client. min_interval throttles high-rate events per type."""
        now = time.time()
        if min_interval:
            last = self._last_emit.get(type_, 0.0)
            if now - last < min_interval:
                return None
        self._last_emit[type_] = now
        msg = {"type": type_, "t": round(now, 3)}
        msg.update(data)
        if type_ not in COALESCE:
            self.history.append(msg)
        with self._lock:
            subs = list(self._subs)
            listeners = list(self._listeners)
        for s in subs:
            s.push(msg)
        for fn in listeners:
            try:
                fn(msg)
            except Exception:
                log.exception("bus listener failed")
        return msg

    # -- consumers -----------------------------------------------------------------
    def subscribe(self, loop: asyncio.AbstractEventLoop) -> Subscriber:
        sub = Subscriber(loop)
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    def listen(self, fn: Callable[[dict], None]) -> Callable[[], None]:
        """Synchronous listener (tests, logging). Returns a function that removes it."""
        with self._lock:
            self._listeners.append(fn)

        def remove():
            with self._lock:
                if fn in self._listeners:
                    self._listeners.remove(fn)
        return remove

    def clients(self) -> int:
        with self._lock:
            return len(self._subs)

    def find(self, type_: str) -> list:
        return [m for m in list(self.history) if m.get("type") == type_]


def dumps(msg: dict) -> str:
    return json.dumps(msg, ensure_ascii=False, default=str)
