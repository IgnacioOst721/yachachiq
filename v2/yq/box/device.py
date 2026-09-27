"""Jetson-side driver of the box ESP32 (protocol: firmware/box_esp32/PROTOCOL.md).

    from yq.box.device import get_box
    box = get_box()                 # real board, or the simulator when mock("box")
    box.rotate_to(90)               # waits for the move to finish
    box.light("rake3", 1.0, max_ms=3000)
    w = box.weigh()                 # {"grams", "sigma_g", "samples", "stable", ...}

One reader thread owns the port (reads lines, matches replies to requests by
id, dispatches events, reconnects after a cable pull); one heartbeat thread
keeps the firmware's 3 s watchdog fed. Requests that are safe to repeat are
retried after a timeout; rotate_by is implemented with an absolute rotate_to.
"""
from __future__ import annotations

import collections
import itertools
import logging
import threading
import time
from typing import Callable, Optional

from yq.common import config

from . import settings as S
from .boxapi import BoxApiMixin
from .ports import candidate_ports
from .protocol import (IDEMPOTENT, SLOW, BoxError, BoxTimeout, BoxUnavailable, LineBuffer, ProtocolError,
                       decode_line, encode_request, raise_for_reply)

log = logging.getLogger("yq.box")


class _Pending:
    __slots__ = ("event", "msg", "error")

    def __init__(self):
        self.event = threading.Event()
        self.msg: Optional[dict] = None
        self.error: Optional[Exception] = None


class Box(BoxApiMixin):
    def __init__(self, port: Optional[str] = None, baud: Optional[int] = None, auto_reconnect: bool = True,
                 heartbeat: bool = True):
        self.port = port or S.SERIAL_PORT or None
        self.baud = baud or S.SERIAL_BAUD
        self.auto_reconnect = auto_reconnect
        self.heartbeat_enabled = heartbeat
        self.sim = None                      # BoxSimulator when simulated
        self.link = None                     # PtyLink when simulated
        self._ser = None
        self._ids = itertools.count(1)
        self._pending: dict = {}
        self._plock = threading.Lock()
        self._wlock = threading.Lock()
        self._listeners: list = []
        self._events = collections.deque(maxlen=500)
        self._econd = threading.Condition()
        self._moves: dict = collections.OrderedDict()   # ref -> move_done event
        self._closing = threading.Event()
        self._connected = threading.Event()
        self._last_tx = 0.0
        self._threads: list = []
        self.resets = 0                      # unexpected ESP32 reboots seen while connected
        self.info_cache: dict = {}
        self.platter_deg = 0.0
        self.target_deg = 0.0

    # -- connection -----------------------------------------------------------------------------
    def connect(self, timeout: Optional[float] = None) -> "Box":
        if self.port is None:
            self.port = self._discover()
        self._open()
        for fn, name in ((self._reader, "box-rx"), (self._heartbeat, "box-hb")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        deadline = time.time() + (timeout if timeout is not None else S.BOOT_WAIT_S + 3.0)
        last: Exception = BoxUnavailable("no answer from %s" % self.port)
        while time.time() < deadline:   # opening the port resets a real DevKitC: wait for it
            try:
                self.info_cache = self.request("info", timeout=1.0, retries=0)
                break
            except (BoxTimeout, BoxUnavailable) as e:
                last = e
        else:
            self.close()
            raise last
        if self.info_cache.get("fw") != "yq-box":
            self.close()
            raise BoxUnavailable("%s is not the Yachachiq box firmware: %r" % (self.port, self.info_cache))
        st = self.status()
        self.platter_deg = self.target_deg = float(st["motor"]["deg"])
        return self

    def _discover(self) -> str:
        ports = candidate_ports()
        if not ports:
            raise BoxUnavailable("no serial port found for the box ESP32 (is the USB cable connected?)")
        if len(ports) == 1:
            return ports[0]
        for p in ports:   # several USB-serial devices: ask each one
            try:
                probe = Box(port=p, auto_reconnect=False, heartbeat=False).connect(timeout=S.BOOT_WAIT_S + 2)
                probe.close()
                return p
            except BoxError:
                continue
        raise BoxUnavailable("none of %s answered as the box firmware" % ports)

    def _open(self) -> None:
        import serial   # lazy: pyserial is only needed on the Jetson
        ser = serial.Serial()
        ser.port, ser.baudrate, ser.timeout, ser.write_timeout = self.port, self.baud, 0.05, 1.0
        # DTR and RTS both released: the DevKitC auto-reset circuit only resets (or enters
        # the bootloader) when they differ, so this avoids a reboot on every connection
        # where the OS allows it. connect() still waits for a possible boot.
        ser.dtr = False
        ser.rts = False
        ser.open()
        self._ser = ser
        self._connected.set()
        log.info("box: serial port %s open", self.port)

    def _drop(self, why: str) -> None:
        self._connected.clear()
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        with self._plock:
            for p in self._pending.values():
                p.error = BoxUnavailable(why)
                p.event.set()
        (log.info if why == "closed" else log.warning)("box: connection %s", "closed" if why == "closed" else "lost (%s)" % why)

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def close(self) -> None:
        self._closing.set()
        self._drop("closed")
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=1.0)
        if self.link is not None:
            self.link.stop()
            self.link = None

    # -- threads --------------------------------------------------------------------------------
    def _reader(self) -> None:
        buf = LineBuffer()
        while not self._closing.is_set():
            ser = self._ser
            if ser is None:
                if not self.auto_reconnect:
                    return
                time.sleep(1.0)
                try:
                    self._open()
                    buf = LineBuffer()
                except Exception as e:
                    log.debug("box: reconnect failed: %s", e)
                continue
            try:
                data = ser.read(ser.in_waiting or 1)
            except Exception as e:
                if not self._closing.is_set():
                    self._drop("read error: %s" % e)
                continue
            for raw in buf.feed(data):
                self._handle_line(raw)

    def _handle_line(self, raw: bytes) -> None:
        try:
            msg = decode_line(raw)
        except ProtocolError as e:
            log.debug("box: ignored line (%s)", e)
            return
        if msg.pop("_kind") == "reply":
            with self._plock:
                p = self._pending.get(msg.get("id"))
            if p is None:
                log.debug("box: late/unknown reply %s", msg)
                return
            p.msg = msg
            p.event.set()
            return
        name = msg.get("event")
        if name == "boot" and self.info_cache:
            self.resets += 1
            log.warning("box: the ESP32 rebooted (%s)", msg.get("reset_reason"))
        if name == "move_done":
            self.platter_deg = float(msg.get("deg", self.platter_deg))
            self._moves[msg.get("ref")] = msg
            while len(self._moves) > 64:
                self._moves.popitem(last=False)
        with self._econd:
            self._events.append((time.time(), msg))
            self._econd.notify_all()
        for fn in list(self._listeners):
            try:
                fn(msg)
            except Exception:
                log.exception("box: event listener failed")

    def _heartbeat(self) -> None:
        while not self._closing.wait(S.HEARTBEAT_S / 2):
            if not self.heartbeat_enabled or not self.connected:
                continue
            if time.time() - self._last_tx >= S.HEARTBEAT_S:
                try:
                    self.request("hb", timeout=1.0, retries=0)
                except BoxError:
                    pass

    # -- requests ---------------------------------------------------------------------------------
    def _write(self, data: bytes) -> None:
        ser = self._ser
        if ser is None:
            raise BoxUnavailable("not connected")
        with self._wlock:
            try:
                ser.write(data)
                ser.flush()
            except Exception as e:
                self._drop("write error: %s" % e)
                raise BoxUnavailable(str(e))
            self._last_tx = time.time()

    def request(self, cmd: str, timeout: Optional[float] = None, retries: Optional[int] = None, **params) -> dict:
        """Send one command and return its reply; raises BoxError (ok:false),
        BoxTimeout or BoxUnavailable."""
        if timeout is None:
            timeout = 12.0 if cmd in SLOW else S.REQUEST_TIMEOUT_S
        if retries is None:
            retries = S.REQUEST_RETRIES if cmd in IDEMPOTENT else 0
        last: BoxError = BoxTimeout("%s: no reply" % cmd)
        for attempt in range(retries + 1):
            if not self._connected.wait(3.0 if attempt else 0.5):
                last = BoxUnavailable("box not connected (%s)" % (self.port or "no port"))
                continue
            rid = next(self._ids)
            slot = _Pending()
            with self._plock:
                self._pending[rid] = slot
            try:
                self._write(encode_request(rid, cmd, **params))
                if slot.event.wait(timeout):
                    if slot.error is not None:
                        last = slot.error
                        continue
                    return raise_for_reply(slot.msg)
                last = BoxTimeout("%s: no reply in %.1f s" % (cmd, timeout))
                log.warning("box: %s timed out (attempt %d)", cmd, attempt + 1)
            except BoxUnavailable as e:
                last = e
            finally:
                with self._plock:
                    self._pending.pop(rid, None)
        raise last

    # -- events -------------------------------------------------------------------------------------
    def add_listener(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def events(self, name: Optional[str] = None, since: float = 0.0) -> list:
        with self._econd:
            return [m for t, m in self._events if t >= since and (name is None or m.get("event") == name)]

    def wait_event(self, name: str, pred: Optional[Callable[[dict], bool]] = None, timeout: float = 5.0,
                   since: Optional[float] = None) -> dict:
        """First buffered event `name` (matching pred) received at or after `since`
        (default: any still in the 500-event buffer). Use since=time.time() taken
        before an action to ignore older events."""
        since = 0.0 if since is None else since
        deadline = time.time() + timeout
        with self._econd:
            while True:
                for t, m in self._events:
                    if t >= since and m.get("event") == name and (pred is None or pred(m)):
                        return m
                left = deadline - time.time()
                if left <= 0:
                    raise BoxTimeout("event %s not received in %.1f s" % (name, timeout))
                self._econd.wait(left)


_box: Optional[Box] = None
_box_lock = threading.Lock()


def get_box(fresh: bool = False) -> Box:
    """The shared Box: the simulator (through a pty) when mock("box"), else the real ESP32."""
    global _box
    with _box_lock:
        if _box is not None and not fresh:
            return _box
        if _box is not None:
            _box.close()
        if config.mock("box"):
            from .sim import BoxSimulator
            from .simlink import PtyLink
            link = PtyLink(BoxSimulator(object_g=S.SIM_OBJECT_G)).start()
            box = Box(port=link.port)
            box.sim, box.link = link.sim, link
            _box = box.connect()
        else:
            _box = Box().connect()
        return _box
