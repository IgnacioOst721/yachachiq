"""Runs the ESP32 simulator behind a pseudo-terminal (macOS / Linux).

    link = PtyLink(BoxSimulator()).start()
    box = Box(port=link.port)          # the real pyserial driver
    ...
    link.stop()

The simulator keeps its own slave fd open so the master never sees a hang-up
when the driver closes and reopens the port (reconnect tests).
"""
from __future__ import annotations

import os
import select
import threading
import time
import tty
from typing import Optional

from .protocol import LineBuffer
from .sim import BoxSimulator


class PtyLink:
    def __init__(self, sim: Optional[BoxSimulator] = None, tick_s: float = 0.002):
        self.sim = sim or BoxSimulator()
        self.tick_s = tick_s
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)          # no echo, no CR/LF translation before pyserial opens it
        tty.setraw(self.master)
        self.port = os.ttyname(self.slave)
        self._stop = threading.Event()
        self._threads: list = []
        self._wlock = threading.Lock()
        self.connected = True           # False = simulate an unplugged cable (bytes are dropped)
        self.rx_lines = 0

    def _write(self, data: bytes) -> None:
        if not self.connected:
            return
        with self._wlock:
            view = memoryview(data)
            while view and not self._stop.is_set():
                try:
                    n = os.write(self.master, view)
                    view = view[n:]
                except BlockingIOError:
                    time.sleep(0.001)
                except OSError:
                    return

    def start(self) -> "PtyLink":
        self.sim.attach(self._write)
        for fn, name in ((self._reader, "sim-rx"), (self._ticker, "sim-tick")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def _reader(self) -> None:
        buf = LineBuffer(max_line=2048)
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([self.master], [], [], 0.05)
                if not r:
                    continue
                data = os.read(self.master, 4096)
            except OSError:
                time.sleep(0.01)
                continue
            if not self.connected:
                continue
            for line in buf.feed(data):
                self.rx_lines += 1
                try:
                    self.sim.handle_line(line)
                except Exception as e:  # a simulator bug must not kill the link
                    self.sim.emit({"event": "fault", "what": "sim", "msg": repr(e)[:200]})

    def _ticker(self) -> None:
        while not self._stop.is_set():
            try:
                self.sim.tick()
            except Exception as e:
                self.sim.emit({"event": "fault", "what": "sim", "msg": repr(e)[:200]})
            time.sleep(self.tick_s)

    def reset_board(self, reason: str = "external") -> None:
        """Like pressing EN: state lost, boot event sent."""
        self.sim.boot(reason)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass
