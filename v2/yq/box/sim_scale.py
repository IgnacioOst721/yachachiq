"""Scale commands of the simulator (mirror of firmware/box_esp32/src/scale.cpp)."""
from __future__ import annotations

from .scalemath import window_stats

DEFAULT_FACTOR = 420.0
MAX_N = 64


class ScaleMixin:
    def _factor_or_default(self) -> float:
        return self.factor if (self.factor is not None and abs(self.factor) > 1) else DEFAULT_FACTOR

    def _start_scale(self, rid, req, t, op):
        if self.scale_op:
            return self._err(rid, "scale_busy")
        if not self.scale.present:
            return self._err(rid, "hx711_missing")
        if self.motor.moving:
            return self._err(rid, "moving")
        n = req.get("n", 20)
        stable_g = req.get("stable_g", 0.5)
        timeout_ms = req.get("timeout_ms", 8000)
        grams = req.get("grams", 0.0)
        if not (isinstance(n, int) and 3 <= n <= MAX_N) or not stable_g > 0 or not 500 <= timeout_ms <= 60000:
            return self._err(rid, "bad_param")
        if op == "cal" and not self.tared:
            return self._err(rid, "not_tared")
        if op == "cal" and not grams > 0:
            return self._err(rid, "bad_param")
        if op == "weigh" and not self.tared:
            return self._err(rid, "not_tared")
        if op == "weigh" and not (self.factor is not None and abs(self.factor) > 1):
            return self._err(rid, "not_calibrated")
        self.scale_op = {"op": op, "id": rid, "n": n, "stable_g": float(stable_g), "grams": float(grams),
                         "t0": t, "timeout_s": timeout_ms / 1000.0, "buf": [], "skip": True}
        self.channels["fan"].held = True
        # drop samples converted before the request (the firmware discards one)
        self.scale.samples_until(t, self.motor.moving, False)
        return None   # reply later

    def _cmd_scale_tare(self, rid, req, t):
        return self._start_scale(rid, req, t, "tare")

    def _cmd_scale_cal(self, rid, req, t):
        return self._start_scale(rid, req, t, "cal")

    def _cmd_weigh(self, rid, req, t):
        return self._start_scale(rid, req, t, "weigh")

    def _end_scale(self):
        self.scale_op = None
        self.channels["fan"].held = False

    def _save_scale(self):
        self.nvs["scale"] = {"offset": self.offset, "tared": self.tared, "factor": self.factor}

    def scale_tick(self, t: float) -> None:
        fan_on = self.channels["fan"].level > 0 and not self.channels["fan"].held
        raws = self.scale.samples_until(t, self.motor.moving, fan_on) if self.scale.present else []
        op = self.scale_op
        if not op:
            return
        for raw in raws:
            if op["skip"]:
                op["skip"] = False
                continue
            op["buf"] = (op["buf"] + [raw])[-MAX_N:]
            if len(op["buf"]) >= op["n"]:
                f = abs(self._factor_or_default())
                st = window_stats(op["buf"][-op["n"]:])
                if st.sigma / f <= op["stable_g"] and st.drift / f <= op["stable_g"]:
                    return self._finish_scale(True)
        if t - op["t0"] > op["timeout_s"]:
            self._finish_scale(False)

    def _finish_scale(self, stable: bool) -> None:
        op = self.scale_op
        rid, buf = op["id"], op["buf"]
        n = min(len(buf), op["n"])
        if n < 3:
            self._err(rid, "scale_timeout", "too few samples")
            return self._end_scale()
        win = buf[-n:]
        st = window_stats(win)
        f = self._factor_or_default()
        if op["op"] == "tare":
            if not stable:
                self._err(rid, "scale_timeout", "not stable")
            else:
                self.offset, self.tared = int(round(st.mean)), True
                self._save_scale()
                self.emit({"id": rid, "ok": True, "offset": self.offset, "sigma_raw": st.sigma, "stable": True})
        elif op["op"] == "cal":
            fac = (st.mean - self.offset) / op["grams"]
            if not stable:
                self._err(rid, "scale_timeout", "not stable")
            elif abs(fac) < 1.0:
                self._err(rid, "bad_param", "no load change: is the weight on the platter?")
            else:
                self.factor = fac
                self._save_scale()
                self.emit({"id": rid, "ok": True, "factor": fac, "raw_mean": st.mean})
        else:
            samples = [round((r - self.offset) / self.factor, 2) for r in win]
            self.emit({"id": rid, "ok": True, "grams": (st.mean - self.offset) / self.factor,
                       "sigma_g": st.sigma / abs(f), "stable": stable, "samples": samples,
                       "tare_raw": self.offset, "tare_g": self.offset / self.factor, "factor": self.factor,
                       "raw_mean": st.mean})
        self._end_scale()

