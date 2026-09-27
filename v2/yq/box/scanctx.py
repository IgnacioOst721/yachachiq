"""Shared state of one scan run: progress, cancel, door waits, virtual time."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from yq.common.contracts import Progress

from .protocol import BoxError

log = logging.getLogger("yq.box.scan")
DOOR_WAIT_S = 90.0


class ScanCancelled(Exception):
    pass


class ScanAborted(Exception):
    """Unrecoverable problem; message_es is shown to the visitor."""

    def __init__(self, message_es: str):
        super().__init__(message_es)
        self.message_es = message_es


class ScanContext:
    def __init__(self, box, folder: Path, on_progress: Optional[Callable] = None,
                 cancel_event: Optional[threading.Event] = None):
        self.box = box
        self.folder = Path(folder)
        self.on_progress = on_progress
        self.cancel_event = cancel_event or threading.Event()
        self.warnings: list = []
        self.cameras_meta: dict = {}
        self.time_scale = float(getattr(box.sim, "time_scale", 1.0) or 1.0) if box.sim is not None else 1.0
        self.scene_fn = box.sim.scene if box.sim is not None else None
        self._span = (0.0, 1.0)

    # -- progress -------------------------------------------------------------------------------------
    def span(self, start: float, end: float) -> None:
        """Following progress(fraction) calls are relative to [start, end] of the whole scan."""
        self._span = (start, end)

    def progress(self, stage: str, local: float, message_es: str, **detail) -> None:
        a, b = self._span
        frac = a + (b - a) * max(0.0, min(1.0, local))
        if self.on_progress:
            try:
                self.on_progress(Progress(stage=stage, fraction=round(frac, 4), message_es=message_es,
                                          detail=detail))
            except Exception:
                log.exception("on_progress failed")

    def warn(self, text_es: str) -> None:
        log.warning("scan: %s", text_es)
        self.warnings.append(text_es)

    # -- time / cancel ------------------------------------------------------------------------------------
    def check(self) -> None:
        if self.cancel_event.is_set():
            raise ScanCancelled()

    def sleep(self, seconds: float) -> None:
        """Wait `seconds` of box time (shorter in real time when the simulator runs faster)."""
        if self.cancel_event.wait(max(0.0, seconds) / self.time_scale):
            raise ScanCancelled()

    # -- doors ------------------------------------------------------------------------------------------------
    def require_doors(self, stage: str) -> None:
        """Block until both doors are closed (asking the visitor), or abort after DOOR_WAIT_S."""
        if self.box.doors_closed():
            return
        self.progress(stage, 0.0, "Cierra las dos puertas de la caja para continuar.", waiting="doors")
        deadline = time.time() + DOOR_WAIT_S
        while time.time() < deadline:
            self.check()
            if self.cancel_event.wait(0.3):
                raise ScanCancelled()
            if self.box.doors_closed():
                self.sleep(0.5)
                return
        raise ScanAborted("Las puertas siguen abiertas; el escaneo se detuvo.")

    def rotate_to(self, deg: float, stage: str) -> dict:
        """rotate_to with a retry when a door opening stopped the move."""
        for attempt in range(3):
            self.check()
            try:
                return self.box.rotate_to(deg)
            except BoxError as e:
                if e.code not in ("stopped", "interlock") or attempt == 2:
                    raise
                self.require_doors(stage)
        raise ScanAborted("El plato no pudo girar.")

    def light(self, ch: str, level: float, max_ms: Optional[int] = None, stage: str = "") -> dict:
        """box.light with door handling (UV/halogen) and cool-down waits."""
        for attempt in range(3):
            self.check()
            try:
                return self.box.light_wait(ch, level, max_ms, max_wait_s=120.0 / self.time_scale + 5)
            except BoxError as e:
                if e.code != "interlock" or attempt == 2:
                    raise
                self.require_doors(stage)
        raise ScanAborted("No se pudo encender %s." % ch)
