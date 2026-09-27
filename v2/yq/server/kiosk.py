"""The kiosk: one visitor flow at a time, the activity timer, status and background jobs."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from yq.common import config
from yq.server import settings
from yq.server.bus import EventBus

log = logging.getLogger("yq.server.kiosk")


class CameraPump:
    """Grabs visitor-camera frames at ~10 fps while the consent screen is open."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[bytes] = None
        self._t = 0.0
        self.camera = None

    def start(self, camera) -> None:
        self.stop()
        self.camera, self._frame = camera, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(camera, self._stop), name="camera-pump", daemon=True)
        self._thread.start()

    def _run(self, camera, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                jpg = camera.grab_jpeg()
            except Exception:
                jpg = None
            if jpg:
                self._frame, self._t = jpg, time.time()
            stop.wait(0.1)

    def latest(self, max_age: float = 2.0) -> Optional[bytes]:
        return self._frame if self._frame and time.time() - self._t < max_age else None

    def stop(self) -> None:
        self._stop.set()
        cam, self.camera = self.camera, None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass

    @property
    def running(self) -> bool:
        return self._thread is not None and not self._stop.is_set()


class Kiosk:
    def __init__(self, bus: Optional[EventBus] = None, subsystems=None, background: bool = True):
        from yq.server.adapters import Subsystems
        self.bus = bus or EventBus()
        self.sub = subsystems or Subsystems()
        self.boot_id = str(int(time.time() * 1000))
        self.flow = None
        self.last_activity = time.time()
        self.pump = CameraPump()
        self.stopping = threading.Event()
        self._lock = threading.Lock()
        self._status: dict = {"mac": None, "printer": None, "hologram": None, "t": 0.0}
        self._background = background
        self.home_data: dict = {}

    # -- background ---------------------------------------------------------------------------
    def start_background(self) -> None:
        if not self._background:
            return
        threading.Thread(target=self._status_loop, name="status", daemon=True).start()
        threading.Thread(target=self._publish_loop, name="publish-loop", daemon=True).start()

    def _status_loop(self) -> None:
        while not self.stopping.is_set():
            self.refresh_status()
            self.stopping.wait(float(settings.STATUS_REFRESH_S))

    def _publish_loop(self) -> None:
        from yq.publish import settings as pset
        from yq.publish import sync as publish
        self.stopping.wait(5.0)
        while not self.stopping.is_set():
            try:
                if publish.mode() == "git" and publish.pending():
                    publish.sync_later(on_done=lambda r: self.bus.emit("published", **r))
            except Exception:
                log.exception("publish loop")
            self.stopping.wait(float(pset.PUBLISH_EVERY_S))

    def refresh_status(self) -> dict:
        from yq.common.macclient import client
        st = {"t": time.time()}
        try:
            st["mac"] = bool(client().available(max_age_s=0))
        except Exception:
            st["mac"] = False
        for name in ("printer", "hologram"):
            try:
                st[name] = bool(getattr(self.sub, name).available(max_age_s=0))
            except Exception:
                st[name] = False
        self._status = st
        self.bus.emit("status", **self.status())
        return st

    def status(self) -> dict:
        from yq.publish import sync as publish
        pub = {}
        try:
            pub = publish.status()
        except Exception as e:
            pub = {"error": str(e)}
        return {"boot_id": self.boot_id, "subsystems": self.sub.modes(),
                "reachable": {k: self._status.get(k) for k in ("mac", "printer", "hologram")},
                "checked": self._status.get("t"), "publish": pub,
                "flow": self.flow.kind if self.flow else None, "screen": self.flow.screen if self.flow else "home",
                "mock": bool(config.MOCK), "clients": self.bus.clients()}

    # -- visitor ------------------------------------------------------------------------------
    def touch(self) -> None:
        self.last_activity = time.time()

    def start(self, kind: str, **opts):
        """Start a story or scan flow. Returns (ok, error_es)."""
        from yq.server.flows.scan import ScanFlow
        from yq.server.flows.story import StoryFlow
        cls = {"story": StoryFlow, "scan": ScanFlow}.get(kind)
        if cls is None:
            return False, "flujo desconocido"
        with self._lock:
            if self.flow is not None and not self.flow.finished.is_set():
                return False, "ocupado"
            self.touch()
            self.flow = cls(self, **opts)
            self.flow.start()
        return True, ""

    def act(self, action: str, payload: Optional[dict] = None) -> bool:
        self.touch()
        f = self.flow
        return bool(f and not f.finished.is_set() and f.act(action, payload or {}))

    def cancel(self, reason: str = "cancel", wait: float = 0.0) -> bool:
        f = self.flow
        if f is None:
            self.show_home()
            return False
        f.cancel(reason)
        if wait:
            f.finished.wait(wait)
        return True

    def flow_ended(self, flow) -> None:
        nxt = None
        with self._lock:
            if self.flow is flow:
                self.flow = None
                nxt = flow.next_flow
        log.info("flow %s ended: %s", flow.kind, flow.outcome)
        self.bus.emit("flow_end", flow=flow.kind, outcome=flow.outcome)
        if nxt and not self.stopping.is_set():
            self.start(nxt)
        else:
            self.show_home(reset=flow.outcome == "idle")

    def show_home(self, reset: bool = False) -> None:
        self.home_data = {"reset": reset}
        self.bus.emit("screen", flow=None, screen="home", data=self.home_data)

    def snapshot(self) -> dict:
        f = self.flow
        scr = {"flow": f.kind, "screen": f.screen, "data": f.data} if f and f.screen else \
            {"flow": None, "screen": "home", "data": self.home_data}
        return {"type": "hello", "boot_id": self.boot_id, "t": time.time(),
                "idle_reset_s": float(config.IDLE_RESET_SECONDS), "ui_language": config.UI_LANGUAGE,
                "mock": bool(config.MOCK), **scr}

    def shutdown(self) -> None:
        self.stopping.set()
        f = self.flow
        if f is not None:
            f.cancel("shutdown")
            f.finished.wait(5)
        self.pump.stop()
