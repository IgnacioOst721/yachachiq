"""Faithful simulator of the box ESP32 firmware (same JSON-lines protocol).

`BoxSimulator` reproduces the firmware logic (firmware/box_esp32/src/*.cpp):
motor timing, the noisy HX711 with settling and the stability rule, door
debounce and the UV/halogen interlock, per-channel on-time limits and
cool-downs, the host heartbeat watchdog and the NVS-persisted settings.
`simlink.PtyLink` runs it behind a pseudo-terminal so the real serial driver
(`yq.box.device.Box`) talks to it exactly like to the hardware.

Test hooks: `set_door()`, `place_object()`, `fail_tmc`, `scale.present`.
"""
from __future__ import annotations

import copy
import threading
import time
from typing import Callable, Optional

from . import settings as S
from .protocol import encode_message
from .sim_cmds import CommandsMixin
from .sim_scale import ScaleMixin
from .simparts import ChannelState, MotorModel, ScaleModel

FW_VERSION = "2.0.0-sim"
DOORS = ("front", "shutter")
CLOSE_DEBOUNCE_S = 0.050


class BoxSimulator(CommandsMixin, ScaleMixin):
    FW_VERSION = FW_VERSION

    def __init__(self, time_scale: Optional[float] = None, calibrated: bool = True, doors_closed: bool = True,
                 object_g: float = 0.0, seed: int = 7):
        self.time_scale = float(time_scale or S.SIM_TIME_SCALE)
        self.lock = threading.RLock()
        self._t0 = time.monotonic()
        self._send: Optional[Callable[[bytes], None]] = None
        self.sent_events: list = []
        self.fail_tmc = False
        self.nvs = {"cfg": copy.deepcopy(S.FIRMWARE_DEFAULTS), "scale": {}}
        self.scale = ScaleModel(seed=seed, object_g=object_g)
        if calibrated:   # as if tare + calibrate-scale had been done once with the empty platter
            self.nvs["scale"] = {"offset": round(self.scale.zero_raw + self.scale.counts_per_g * self.scale.platter_g),
                                 "tared": True, "factor": self.scale.counts_per_g * 1.0004}
        self.door_raw = {d: bool(doors_closed) for d in DOORS}
        self.boot("poweron", emit=False)
        for d in DOORS:  # doors already settled at power-on
            self.door_closed[d] = self.door_raw[d]

    # -- time ---------------------------------------------------------------------------------
    def now(self) -> float:
        """Virtual seconds since start (real seconds x time_scale)."""
        return (time.monotonic() - self._t0) * self.time_scale

    def real_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    # -- output ---------------------------------------------------------------------------------
    def attach(self, send: Callable[[bytes], None]) -> None:
        self._send = send

    def emit(self, msg: dict) -> None:
        if "event" in msg:
            self.sent_events.append(msg)
        if self._send:
            self._send(encode_message(msg))

    def event(self, name: str, **fields) -> None:
        self.emit({"event": name, **fields})

    # -- lifecycle ------------------------------------------------------------------------------
    def boot(self, reason: str = "poweron", emit: bool = True) -> None:
        with self.lock:
            self.cfg = copy.deepcopy(self.nvs["cfg"])
            spr = int(round(self.cfg["motor_steps"] * self.cfg["microsteps"] * self.cfg["ratio"]))
            self.motor = MotorModel(spr, self.cfg["dps"], self.cfg["accel"])
            self.move_ref = -1
            self.move_stopped = False
            self.channels = {c: ChannelState(c, S.CHANNEL_KIND[c]) for c in S.CHANNELS}
            self.door_closed = {d: False for d in DOORS}
            self.low_since = {d: None for d in DOORS}
            self.hb_armed, self.hb_lost, self.hb_last_ms = False, False, 0
            sc = self.nvs["scale"]
            self.offset, self.tared, self.factor = sc.get("offset", 0), sc.get("tared", False), sc.get("factor")
            self.scale_op = None
            self.tmc_ok = not self.fail_tmc
            if emit:
                self.event("boot", fw="yq-box", version=FW_VERSION, proto=1, reset_reason=reason)
                if not self.tmc_ok:
                    self.event("fault", what="tmc_uart", msg="TMC2209 not answering on UART")

    # -- test hooks -----------------------------------------------------------------------------
    def set_door(self, door: str, closed: bool) -> float:
        """Change a reed input. Opening cuts UV/halogen synchronously (the
        firmware's GPIO interrupt). Returns the virtual time of the change."""
        with self.lock:
            self.door_raw[door] = bool(closed)
            t = self.now()
            if not closed:
                self._interlock_cut(t)
                self.low_since[door] = None
                if self.door_closed[door]:
                    self.door_closed[door] = False
                    self.event("door", door=door, closed=False)
                    if self.cfg["door_stops_motor"]:
                        self.motor.stop(t, hard=False)
                        self.move_stopped = self.move_stopped or self.motor.moving
            return t

    def place_object(self, grams: float) -> None:
        with self.lock:
            self.scale.place(grams)

    def scene(self) -> dict:
        """What the mock cameras see."""
        with self.lock:
            t = self.now()
            return {"platter_deg": self.motor.deg(t), "moving": self.motor.moving,
                    "lights": {c: (ch.level if not ch.held else 0.0) for c, ch in self.channels.items()},
                    "doors_closed": all(self.door_raw.values()), "t": t, "object_g": self.scale.object_g,
                    "time_scale": self.time_scale}

    # -- safety ---------------------------------------------------------------------------------
    def doors_ok(self) -> bool:
        return all(self.door_closed.values()) and all(self.door_raw.values())

    def _interlock_cut(self, t: float) -> None:
        cut = [c for c in ("uv", "halogen") if self.channels[c].level > 0]
        for c in cut:
            self._turn_off(c, t, "interlock")
        if cut:
            self.event("interlock", cut=cut)

    def _turn_off(self, name: str, t: float, reason: Optional[str]) -> bool:
        ch = self.channels[name]
        if ch.level <= 0:
            return False
        on_for = t - ch.on_since
        ch.level = 0.0
        f = self.cfg["cool_factor"][ch.kind]
        if f > 0:
            ch.cool_until = t + f * on_for
        if reason:
            self.event("light_off", ch=name, reason=reason, on_ms=int(on_for * 1000))
        return True

    def all_off(self, reason: Optional[str]) -> list:
        t = self.now()
        return [c for c in S.CHANNELS if self._turn_off(c, t, reason)]

    # -- periodic work (the firmware loop) ------------------------------------------------------------
    def tick(self) -> None:
        with self.lock:
            t = self.now()
            for d in DOORS:
                if self.door_raw[d]:
                    if self.low_since[d] is None:
                        self.low_since[d] = t
                    if not self.door_closed[d] and t - self.low_since[d] >= CLOSE_DEBOUNCE_S:
                        self.door_closed[d] = True
                        self.event("door", door=d, closed=True)
            if not self.doors_ok():
                self._interlock_cut(t)
            real = self.real_ms()
            if self.hb_armed and not self.hb_lost and real - self.hb_last_ms > self.cfg["hb_timeout_ms"]:
                self.hb_lost = True
                self.all_off("watchdog")
                self.motor.stop(t, hard=True)
                self.motor.enabled = False
                self.event("watchdog", reason="heartbeat")
            for name, ch in self.channels.items():
                if ch.level > 0 and ch.limit_s > 0 and t - ch.on_since >= ch.limit_s:
                    self._turn_off(name, t, "max_on")
            if self.motor.update(t):
                self.event("move_done", ref=self.move_ref, deg=self.motor.deg(t), steps=self.motor.steps(t),
                           stopped=self.move_stopped)
            self.scale_tick(t)
