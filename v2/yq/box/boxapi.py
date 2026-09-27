"""High-level commands of the box driver (mixed into yq.box.device.Box)."""
from __future__ import annotations

import time
from typing import Optional

from . import settings as S
from .protocol import BoxError, BoxTimeout


class BoxApiMixin:
    # -- basics -------------------------------------------------------------------------------------
    def ping(self) -> dict:
        return self.request("ping")

    def info(self) -> dict:
        self.info_cache = self.request("info")
        return self.info_cache

    def status(self) -> dict:
        st = self.request("status")
        self.last_status = st
        if not st["motor"]["moving"]:
            self.platter_deg = float(st["motor"]["deg"])
        return st

    def doors_closed(self) -> bool:
        return bool(self.status()["doors_closed"])

    # -- motion ------------------------------------------------------------------------------------------
    def motor(self, enable: bool) -> dict:
        return self.request("motor", enable=bool(enable))

    def speed(self, dps: Optional[float] = None, accel: Optional[float] = None) -> dict:
        return self.request("speed", dps=dps, accel=accel)

    def rotate_to(self, deg: float, wait: bool = True, wrap: bool = False, timeout: Optional[float] = None) -> dict:
        """Absolute turntable angle (unwrapped degrees, + = counter-clockwise from above).
        With wait=True returns the move_done event; raises BoxError("stopped") if
        the move was cut short (door opened, stop, watchdog)."""
        since = time.time()
        r = self.request("rotate_to", deg=float(deg), wrap=bool(wrap) or None)
        # keep the exact requested angle (not the step-quantized one) so that
        # rotate_by(10) x 36 lands on 360 without accumulating rounding
        self.target_deg = float(deg) if not wrap else float(r["target_deg"])
        if not wait:
            return r
        done = self.wait_move(r["id"], timeout or r.get("eta_ms", 0) / 1000.0 * 1.5 + 5.0, since)
        if done.get("stopped"):
            raise BoxError("stopped", "move interrupted at %.2f deg" % done.get("deg", 0.0), done)
        return done

    def rotate_by(self, deg: float, wait: bool = True, timeout: Optional[float] = None) -> dict:
        """Relative move, sent as an absolute target so a retry can never double it."""
        return self.rotate_to(self.target_deg + float(deg), wait=wait, timeout=timeout)

    def wait_move(self, ref: int, timeout: float, since: Optional[float] = None) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            done = self._moves.get(ref)
            if done is not None:
                return done
            try:
                return self.wait_event("move_done", lambda m: m.get("ref") == ref,
                                       timeout=min(0.5, max(0.01, deadline - time.time())),
                                       since=since if since is not None else 0.0)
            except BoxTimeout:
                continue
        raise BoxTimeout("move %s did not finish in %.1f s" % (ref, timeout))

    def stop(self, hard: bool = False) -> dict:
        return self.request("stop", hard=bool(hard) or None)

    def zero(self) -> dict:
        r = self.request("zero")
        self.platter_deg = self.target_deg = 0.0
        return r

    # -- scale -------------------------------------------------------------------------------------------
    def tare(self, n: int = 20, timeout_ms: int = None) -> dict:
        t = timeout_ms or S.WEIGH_TIMEOUT_MS
        return self.request("scale_tare", timeout=t / 1000.0 + 4.0, n=n, timeout_ms=t)

    def calibrate_scale(self, grams: float, n: int = 20, timeout_ms: int = None) -> dict:
        t = timeout_ms or S.WEIGH_TIMEOUT_MS
        return self.request("scale_cal", timeout=t / 1000.0 + 4.0, grams=float(grams), n=n, timeout_ms=t)

    def weigh(self, n: int = None, stable_g: float = None, timeout_ms: int = None) -> dict:
        t = timeout_ms or S.WEIGH_TIMEOUT_MS
        return self.request("weigh", timeout=t / 1000.0 + 4.0, n=n or S.WEIGH_SAMPLES,
                            stable_g=stable_g or S.WEIGH_STABLE_G, timeout_ms=t)

    # -- lights --------------------------------------------------------------------------------------------
    def light(self, ch: str, level: float, max_ms: Optional[int] = None) -> dict:
        return self.request("light", ch=ch, level=float(level), max_ms=int(max_ms) if max_ms else None)

    def light_wait(self, ch: str, level: float, max_ms: Optional[int] = None, max_wait_s: float = 120.0) -> dict:
        """light(), waiting out a cool-down if the firmware asks for one."""
        deadline = time.time() + max_wait_s
        while True:
            try:
                return self.light(ch, level, max_ms)
            except BoxError as e:
                wait = (e.reply or {}).get("wait_ms")
                if e.code != "cooldown" or not wait or time.time() + wait / 1000.0 > deadline:
                    raise
                time.sleep(wait / 1000.0 + 0.02)

    def all_off(self) -> dict:
        return self.request("all_off")

    def estop(self) -> dict:
        return self.request("estop")

    # -- config --------------------------------------------------------------------------------------------
    def config_get(self) -> dict:
        return self.request("config_get")["config"]

    def config_set(self, save: bool = False, **values) -> dict:
        return self.request("config_set", save=bool(save) or None, **values)["config"]

    def safe_off(self) -> None:
        """Best effort: everything off and the motor stopped (cancel / error paths)."""
        for fn in (lambda: self.all_off(), lambda: self.stop(hard=False)):
            try:
                fn()
            except BoxError:
                pass
