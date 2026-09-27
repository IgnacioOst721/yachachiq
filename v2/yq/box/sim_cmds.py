"""Request handling of the simulator (mirror of firmware/box_esp32/src/commands.cpp)."""
from __future__ import annotations

import json
import math

from . import settings as S
from .protocol import MAX_LINE


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class CommandsMixin:
    # -- entry point ------------------------------------------------------------------------------
    def handle_line(self, line: bytes) -> None:
        if len(line) > MAX_LINE:
            return self.emit({"id": -1, "ok": False, "error": "line_too_long"})
        try:
            req = json.loads(line.decode("utf-8"))
            if not isinstance(req, dict):
                raise ValueError("not an object")
        except ValueError as e:
            return self.emit({"id": -1, "ok": False, "error": "bad_json", "msg": str(e)[:60]})
        with self.lock:
            rid = req.get("id", -1)
            self.hb_armed, self.hb_lost, self.hb_last_ms = True, False, self.real_ms()
            cmd = req.get("cmd", "")
            fn = getattr(self, "_cmd_" + str(cmd), None)
            if fn is None:
                return self._err(rid, "unknown_cmd", str(cmd))
            reply = fn(rid, req, self.now())
            if reply is not None:
                self.emit({"id": rid, "ok": True, **reply})

    def _err(self, rid, code, msg=None, **extra):
        m = {"id": rid, "ok": False, "error": code, **extra}
        if msg:
            m["msg"] = msg
        self.emit(m)

    # -- simple commands ------------------------------------------------------------------------------
    def _cmd_ping(self, rid, req, t):
        return {"pong": True, "uptime_ms": self.real_ms()}

    def _cmd_hb(self, rid, req, t):
        return {"uptime_ms": self.real_ms()}

    def _cmd_info(self, rid, req, t):
        return {"fw": "yq-box", "version": self.FW_VERSION, "proto": 1,
                "board": "sim", "tmc": {"ok": self.tmc_ok, "version": 0x21 if self.tmc_ok else 0},
                "hx711": self.scale.present, "steps_per_rev": self.motor.spr, "channels": list(S.CHANNELS),
                "reset_reason": "poweron", "sim": True, "time_scale": self.time_scale}

    def _cmd_status(self, rid, req, t):
        lights = {}
        for name, ch in self.channels.items():
            on = (t - ch.on_since) if ch.level > 0 else 0.0
            left = max(0.0, ch.limit_s - on) if ch.level > 0 and ch.limit_s > 0 else 0.0
            lights[name] = {"level": ch.level, "on_ms": int(on * 1000), "left_ms": int(left * 1000),
                            "cool_ms": int(max(0.0, ch.cool_until - t) * 1000)}
        m = self.motor
        cal = self.factor is not None and abs(self.factor) > 1
        faults = ([] if self.tmc_ok else ["tmc_uart"]) + ([] if self.scale.present else ["hx711"])
        return {"uptime_ms": self.real_ms(), "hb_age_ms": self.real_ms() - self.hb_last_ms, "hb_lost": self.hb_lost,
                "doors": {d: self.door_closed[d] for d in self.door_closed}, "doors_closed": self.doors_ok(),
                "motor": {"enabled": m.enabled, "moving": m.moving, "steps": m.steps(t), "deg": m.deg(t),
                          "target_deg": m.target if m.moving else m.deg(t), "dps": m.dps, "accel": m.accel,
                          "tmc_ok": self.tmc_ok, "steps_per_rev": m.spr},
                "lights": lights,
                "scale": {"busy": self.scale_op is not None, "tared": self.tared, "calibrated": cal,
                          "factor": self.factor if cal else None, "offset": self.offset, "hx711": self.scale.present},
                "faults": faults}

    def _cmd_motor(self, rid, req, t):
        if not isinstance(req.get("enable"), bool):
            return self._err(rid, "bad_param", "enable: bool")
        if not req["enable"] and self.motor.moving:
            self.motor.stop(t, hard=True)
            self.move_stopped = True
        self.motor.enabled = req["enable"]
        return {"enabled": self.motor.enabled}

    def _move(self, rid, deg, t):
        if self.cfg["door_stops_motor"] and not self.doors_ok():
            return self._err(rid, "interlock", "door open")
        if not self.tmc_ok and not self.cfg["allow_no_uart"]:
            return self._err(rid, "tmc_uart")
        if not _num(deg) or abs(deg) > 36000:
            return self._err(rid, "bad_param")
        self.motor.enabled = True
        eta = self.motor.start(deg, t)
        self.move_ref, self.move_stopped = rid, False
        # eta is virtual time; the host waits in real time
        self.target_req = float(deg)       # like the firmware: the requested (unquantized) angle
        return {"target_deg": float(deg), "eta_ms": int(eta * 1000 / self.time_scale)}

    def _cmd_rotate_to(self, rid, req, t):
        if not _num(req.get("deg")):
            return self._err(rid, "bad_param", "deg: number")
        if self.scale_op:
            return self._err(rid, "scale_busy")
        deg = float(req["deg"])
        if req.get("wrap"):
            cur = self.motor.deg(t)
            d = math.fmod(deg - cur, 360.0)
            d = d - 360 if d > 180 else (d + 360 if d < -180 else d)
            deg = cur + d
        return self._move(rid, deg, t)

    def _cmd_rotate_by(self, rid, req, t):
        if not _num(req.get("deg")):
            return self._err(rid, "bad_param", "deg: number")
        if self.scale_op:
            return self._err(rid, "scale_busy")
        base = getattr(self, "target_req", self.motor.target) if self.motor.moving else self.motor.deg(t)
        return self._move(rid, base + float(req["deg"]), t)

    def _cmd_speed(self, rid, req, t):
        for k in ("dps", "accel"):
            if k in req and not _num(req[k]):
                return self._err(rid, "bad_param")
        c = self.cfg
        c["dps"] = min(max(float(req.get("dps", c["dps"])), 0.5), c["max_dps"])
        c["accel"] = min(max(float(req.get("accel", c["accel"])), 0.5), c["max_accel"])
        self.motor.dps, self.motor.accel = c["dps"], c["accel"]
        return {"dps": c["dps"], "accel": c["accel"]}

    def _cmd_stop(self, rid, req, t):
        if self.motor.moving:
            self.motor.stop(t, hard=bool(req.get("hard")))
            self.move_stopped = True
        return {"deg": self.motor.deg(t)}

    def _cmd_zero(self, rid, req, t):
        if self.motor.moving:
            return self._err(rid, "moving")
        self.motor.zero()
        return {"deg": 0.0}

    def _cmd_all_off(self, rid, req, t):
        return {"cut": self.all_off(None)}

    def _cmd_estop(self, rid, req, t):
        cut = self.all_off(None)
        self.motor.stop(t, hard=True)
        self.motor.enabled = False
        if self.scale_op:
            self._err(self.scale_op["id"], "scale_timeout", "aborted")
            self._end_scale()
        return {"cut": cut}

    def _cmd_reboot(self, rid, req, t):
        self.emit({"id": rid, "ok": True})
        self.boot("software")
        return None

    # -- lights ---------------------------------------------------------------------------------------
    def _cmd_light(self, rid, req, t):
        name, level = req.get("ch"), req.get("level")
        if name not in self.channels or not _num(level) or not 0 <= level <= 1:
            return self._err(rid, "bad_param", "ch, level")
        ch = self.channels[name]
        if level == 0:
            self._turn_off(name, t, None)
            return {"ch": name, "level": 0.0, "off_in_ms": 0}
        if ch.kind in ("uv", "halogen"):
            if not self.doors_ok():
                return self._err(rid, "interlock")
            level = 1.0
        max_ms = req.get("max_ms") or 0
        if ch.level <= 0:
            if ch.cool_until > t:
                return self._err(rid, "cooldown", wait_ms=int((ch.cool_until - t) * 1000 / self.time_scale))
            if ch.kind == "rake":
                for other in S.RAKE_CHANNELS:
                    if other != name:
                        self._turn_off(other, t, "exclusive")
            lim = self.cfg["max_on_ms"][ch.kind]
            if max_ms and (lim == 0 or max_ms < lim):
                lim = max_ms
            ch.on_since, ch.limit_s = t, lim / 1000.0
        elif max_ms:
            elapsed = t - ch.on_since
            if ch.limit_s == 0 or elapsed + max_ms / 1000.0 < ch.limit_s:
                ch.limit_s = elapsed + max_ms / 1000.0
        ch.level = float(level)
        left = (ch.limit_s - (t - ch.on_since)) if ch.limit_s > 0 else 0.0
        return {"ch": name, "level": ch.level, "off_in_ms": int(left * 1000 / self.time_scale)}

    # -- config ---------------------------------------------------------------------------------------
    def _cmd_config_get(self, rid, req, t):
        c = dict(self.cfg)
        c["steps_per_rev"] = self.motor.spr
        return {"config": c}

    def _cmd_config_set(self, rid, req, t):
        new = self._clamped_config(req)
        if isinstance(new, str):
            return self._err(rid, "bad_param", new)
        geometry = ("motor_steps", "microsteps", "ratio", "run_ma", "hold_pct")
        reboot = any(new[k] != self.cfg[k] for k in geometry)
        if req.get("save"):
            self.nvs["cfg"] = dict(new)
        for k in geometry:
            new[k] = self.cfg[k]
        self.cfg = new
        self.motor.dps, self.motor.accel = new["dps"], new["accel"]
        out = self._cmd_config_get(rid, req, t)
        if reboot:
            out["reboot_required"] = True
        return out

    def _clamped_config(self, req):
        c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.cfg.items()}
        for group in ("max_on_ms", "cool_factor"):
            for kind, v in (req.get(group) or {}).items():
                if kind not in c[group] or not _num(v) or v < 0:
                    return group
                c[group][kind] = v
        for k in ("hb_timeout_ms", "dps", "accel", "max_dps", "max_accel", "motor_steps", "microsteps", "ratio",
                  "run_ma", "hold_pct"):
            if k in req:
                if not _num(req[k]):
                    return k
                c[k] = req[k]
        for k in ("door_stops_motor", "allow_no_uart"):
            if k in req:
                c[k] = bool(req[k])
        for kind, cap in S.HARD_MAX_ON_MS.items():
            if cap and (c["max_on_ms"][kind] == 0 or c["max_on_ms"][kind] > cap):
                c["max_on_ms"][kind] = cap
            c["cool_factor"][kind] = min(max(c["cool_factor"][kind], S.HARD_MIN_COOL[kind]), 20.0)
        c["hb_timeout_ms"] = int(min(max(c["hb_timeout_ms"], 500), 10000))
        c["max_dps"] = min(max(c["max_dps"], 1.0), 120.0)
        c["max_accel"] = min(max(c["max_accel"], 1.0), 360.0)
        c["dps"] = min(max(c["dps"], 0.5), c["max_dps"])
        c["accel"] = min(max(c["accel"], 0.5), c["max_accel"])
        c["motor_steps"] = int(c["motor_steps"]) if c["motor_steps"] in (200, 400) else 200
        c["microsteps"] = int(c["microsteps"]) if c["microsteps"] in (8, 16, 32, 64) else 16
        c["ratio"] = min(max(float(c["ratio"]), 1.0), 100.0)
        c["run_ma"] = int(min(max(c["run_ma"], 300), 1400))
        c["hold_pct"] = int(min(max(c["hold_pct"], 0), 60))
        return c
