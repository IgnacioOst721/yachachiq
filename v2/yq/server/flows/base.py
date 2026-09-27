"""Flow = one visitor experience (story or scan) running as a state machine in a thread.

Each state is a method s_<name>() returning the next state name (or None = finished).
States that wait for the visitor call self.wait(...), which blocks on an action queue,
watches the idle timer and raises Cancelled / IdleTimeout. Slow calls into other
domains go through self.call(...), which runs them in a helper thread so a hung
subsystem can never freeze the kiosk: cancel and timeouts always win.
Any exception becomes a kind Spanish error screen with "Intentar otra vez".
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

from yq.common import config
from yq.server import settings

log = logging.getLogger("yq.server.flow")


class Cancelled(Exception):
    pass


class IdleTimeout(Exception):
    pass


class FlowError(Exception):
    """An error to show to the visitor. key = i18n key in static/i18n/*.json (message_es is the fallback)."""

    def __init__(self, message_es: str, key: str = "", retry: Optional[str] = None, detail: str = ""):
        super().__init__(message_es)
        self.message_es, self.key, self.retry, self.detail = message_es, key, retry, detail


def kind_error(e: BaseException) -> FlowError:
    if isinstance(e, FlowError):
        return e
    name = type(e).__name__
    if name == "MacUnavailable":
        return FlowError("El computador que ayuda al robot no responde. Probemos otra vez.", "err_mac", detail=str(e))
    if name in ("PrinterError",):
        return FlowError("La impresora no responde.", "err_printer", detail=str(e))
    if isinstance(e, TimeoutError):
        return FlowError("Esto está tardando demasiado. Probemos otra vez.", "err_timeout", detail=str(e))
    return FlowError("Algo no salió bien. Puedes intentarlo otra vez.", "err_generic", detail="%s: %s" % (name, e))


class Flow:
    kind = "base"
    first = ""
    retry_map: dict = {}           # state that failed -> state to go back to on "retry"

    def __init__(self, kiosk):
        self.k = kiosk
        self.bus = kiosk.bus
        self.sub = kiosk.sub
        self.cancel_event = threading.Event()
        self.cancel_reason = ""
        self.actions: queue.Queue = queue.Queue()
        self.accepts: set = set()
        self.immediate: dict = {}   # action -> handler(payload), run at once in the request thread
        self.screen, self.data = "", {}
        self.outcome = ""           # done | cancelled | home | idle | error
        self.next_flow: Optional[str] = None
        self.finished = threading.Event()
        self._warned = False
        self.thread = threading.Thread(target=self._main, name="flow-" + self.kind, daemon=True)

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        self.thread.start()

    def cancel(self, reason: str = "cancel") -> None:
        if not self.cancel_event.is_set():
            self.cancel_reason = reason
            self.cancel_event.set()
            try:
                self.on_cancel()
            except Exception:
                log.exception("on_cancel failed")

    def on_cancel(self) -> None:
        """Stop whatever runs right now (recorder, TTS, camera...). Subclasses override."""

    def cleanup(self) -> None:
        """Always called when the flow ends. Subclasses release hardware here."""

    def _main(self) -> None:
        state = self.first
        try:
            while state:
                if self.cancel_event.is_set():
                    raise Cancelled()
                log.info("%s -> %s", self.kind, state)
                try:
                    state = getattr(self, "s_" + state)()
                except (Cancelled, IdleTimeout):
                    raise
                except Exception as e:
                    if self.cancel_event.is_set():
                        raise Cancelled()
                    state = self._error(state, e)
            self.outcome = self.outcome or "done"
        except Cancelled:
            self.outcome = self.cancel_reason or "cancelled"
        except IdleTimeout:
            self.outcome = "idle"
        except Exception:
            log.exception("flow crashed")
            self.outcome = "error"
        finally:
            try:
                self.cleanup()
            except Exception:
                log.exception("cleanup failed")
            self.finished.set()
            self.k.flow_ended(self)

    def _error(self, state: str, e: Exception) -> Optional[str]:
        fe = kind_error(e)
        log.warning("%s: error in %s: %s", self.kind, state, fe.detail or fe.message_es,
                    exc_info=not isinstance(e, FlowError))
        self.show("error", message_es=fe.message_es, key=fe.key, detail=fe.detail[:300], failed=state)
        name, _ = self.wait("retry", "home")
        return fe.retry or self.retry_map.get(state, state) if name == "retry" else None

    # -- screens -------------------------------------------------------------------------
    def show(self, screen: str, **data) -> None:
        self.screen, self.data = screen, data
        self.bus.emit("screen", flow=self.kind, screen=screen, data=data)

    def update(self, **data) -> None:
        self.data.update(data)
        self.bus.emit("screen_update", flow=self.kind, screen=self.screen, data=data)

    def toast(self, text_es: str, key: str = "", kind: str = "info") -> None:
        self.bus.emit("toast", text_es=text_es, key=key, kind=kind)

    # -- visitor actions ------------------------------------------------------------------
    def act(self, name: str, payload: dict) -> bool:
        if name in self.immediate:
            self.immediate[name](payload or {})
            return True
        if name in self.accepts or name == "home":
            self.actions.put((name, payload or {}))
            return True
        return False

    def wait(self, *allowed: str, idle: bool = True, poll: Optional[Callable[[], None]] = None,
             poll_s: float = 0.2, timeout: Optional[float] = None):
        """Block until one of `allowed` actions (or "home") arrives. Returns (name, payload)."""
        while not self.actions.empty():                  # drop taps meant for the previous screen
            self.actions.get_nowait()
        self.accepts = set(allowed)
        t0 = time.time()
        try:
            while True:
                if self.cancel_event.is_set():
                    raise Cancelled()
                try:
                    name, payload = self.actions.get(timeout=poll_s)
                    if name == "home":
                        self.cancel_reason = "home"
                        raise Cancelled()
                    if name in self.accepts:
                        return name, payload
                except queue.Empty:
                    pass
                if poll:
                    poll()
                if idle:
                    self._check_idle()
                if timeout is not None and time.time() - t0 > timeout:
                    return "timeout", {}
        finally:
            self.accepts = set()
            if self._warned:
                self._warned = False
                self.bus.emit("idle_clear")

    def _check_idle(self) -> None:
        quiet = time.time() - self.k.last_activity
        limit = float(config.IDLE_RESET_SECONDS)
        warn_at = max(0.0, limit - float(settings.IDLE_WARNING_SECONDS))
        if quiet >= limit:
            raise IdleTimeout()
        if quiet >= warn_at and not self._warned:
            self._warned = True
            self.bus.emit("idle_warning", seconds=round(limit - quiet))
        elif quiet < warn_at and self._warned:
            self._warned = False
            self.bus.emit("idle_clear")

    def pause(self, seconds: float) -> None:
        """Short pause for the visitor to read something (scaled by YQ_PAUSE_SCALE in tests)."""
        if self.cancel_event.wait(seconds * float(settings.PAUSE_SCALE)):
            raise Cancelled()

    # -- slow calls ------------------------------------------------------------------------
    def call(self, fn: Callable, *args, timeout: Optional[float] = None, on_cancel: Optional[Callable] = None,
             **kw):
        """Run fn in a helper thread; cancel/timeout return control at once (the thread is abandoned)."""
        box: dict = {}
        done = threading.Event()

        def run():
            try:
                box["value"] = fn(*args, **kw)
            except BaseException as e:           # noqa: BLE001 - re-raised in the flow thread
                box["error"] = e
            finally:
                done.set()

        threading.Thread(target=run, name="call-" + getattr(fn, "__name__", "fn"), daemon=True).start()
        t0 = time.time()
        while not done.wait(0.1):
            if self.cancel_event.is_set():
                if on_cancel:
                    try:
                        on_cancel()
                    except Exception:
                        log.exception("on_cancel failed")
                raise Cancelled()
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError("%s took more than %.0f s" % (getattr(fn, "__name__", "call"), timeout))
        if "error" in box:
            raise box["error"]
        return box.get("value")
