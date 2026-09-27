"""Physical models used by the ESP32 simulator (yq.box.sim).

Times are "virtual" seconds (real seconds x SIM_TIME_SCALE) so a full scan can
be simulated faster than real time while keeping the same relative timings.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


class MotorModel:
    """Trapezoidal (or triangular) move profile, like FastAccelStepper."""

    def __init__(self, steps_per_rev: int, dps: float, accel: float):
        self.spr = steps_per_rev
        self.dps = dps
        self.accel = accel
        self.enabled = False
        self._p0 = 0.0          # deg at segment start
        self._v0 = 0.0          # signed deg/s at segment start
        self._t0 = 0.0
        self._target = 0.0
        self._decel_only = False
        self.moving = False

    # -- profile -----------------------------------------------------------------------
    def _state(self, t: float):
        """(position_deg, velocity_deg_s, finished) at virtual time t."""
        if not self.moving:
            return self._p0, 0.0, True
        dt = max(0.0, t - self._t0)
        a = self.accel
        if self._decel_only:
            sgn = 1.0 if self._v0 >= 0 else -1.0
            v = abs(self._v0)
            t_end = v / a
            if dt >= t_end:
                return self._p0 + sgn * v * v / (2 * a), 0.0, True
            return self._p0 + sgn * (v * dt - 0.5 * a * dt * dt), sgn * (v - a * dt), False
        d = self._target - self._p0
        sgn = 1.0 if d >= 0 else -1.0
        d = abs(d)
        v = self.dps
        if d < v * v / a:                       # triangular
            t_half = math.sqrt(d / a)
            t_acc, t_cruise, v = t_half, 0.0, a * t_half
        else:
            t_acc = v / a
            t_cruise = (d - v * v / a) / v
        total = 2 * t_acc + t_cruise
        if dt >= total:
            return self._target, 0.0, True
        if dt < t_acc:
            s, vel = 0.5 * a * dt * dt, a * dt
        elif dt < t_acc + t_cruise:
            s, vel = 0.5 * a * t_acc ** 2 + v * (dt - t_acc), v
        else:
            td = total - dt
            s, vel = d - 0.5 * a * td * td, a * td
        return self._p0 + sgn * s, sgn * vel, False

    def eta_s(self, delta_deg: float) -> float:
        d, v, a = abs(delta_deg), self.dps, self.accel
        return 2 * math.sqrt(d / a) if d < v * v / a else d / v + v / a

    def quantize(self, deg: float) -> float:
        return round(deg * self.spr / 360.0) * 360.0 / self.spr

    def steps(self, t: float) -> int:
        return int(round(self._state(t)[0] * self.spr / 360.0))

    def deg(self, t: float) -> float:
        return self.steps(t) * 360.0 / self.spr

    def start(self, target_deg: float, t: float) -> float:
        """Begin a move (from rest or re-targeting). Returns ETA seconds."""
        p, _v, _done = self._state(t)
        self._p0, self._t0 = p, t
        self._target = self.quantize(target_deg)
        self._decel_only = False
        self.moving = True
        return self.eta_s(self._target - p)

    def stop(self, t: float, hard: bool) -> None:
        if not self.moving:
            return
        p, v, done = self._state(t)
        if hard or done:
            self._p0, self.moving = p, False
            return
        self._p0, self._v0, self._t0, self._decel_only = p, v, t, True

    def update(self, t: float) -> bool:
        """Advance; returns True exactly when a move has just finished."""
        if not self.moving:
            return False
        p, _v, done = self._state(t)
        if done:
            self._p0, self.moving, self._decel_only = p, False, False
            return True
        return False

    def zero(self) -> None:
        self._p0 = 0.0
        self._target = 0.0

    @property
    def target(self) -> float:
        return self._target


@dataclass
class ScaleModel:
    """HX711 + 5 kg cell: 10 samples/s, Gaussian noise, settling after a load
    change, extra noise while the motor turns or the fan blows."""
    zero_raw: float = 83000.0            # ADC reading with nothing on the cell
    counts_per_g: float = -412.3         # sign depends on the bridge wiring
    platter_g: float = 1150.0            # tare: platter + bearing + crown (CAD R1 estimate)
    noise_g: float = 0.12
    motor_noise_g: float = 25.0
    fan_noise_g: float = 2.5
    tau_s: float = 0.5                   # settling time constant after a load change
    rate_hz: float = 10.0
    object_g: float = 0.0
    present: bool = True
    seed: int = 7
    _shown_g: float = field(default=None, init=False)
    _last_t: float = field(default=0.0, init=False)
    _next_sample_t: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._shown_g = self.platter_g + self.object_g

    def place(self, grams: float) -> None:
        self.object_g = float(grams)

    def samples_until(self, t: float, motor_moving: bool, fan_on: bool) -> list:
        """Raw ADC samples produced between the last call and virtual time t."""
        out = []
        if self._next_sample_t == 0.0:
            self._next_sample_t = t
        while self._next_sample_t <= t:
            ts = self._next_sample_t
            dt = ts - self._last_t if self._last_t else 1.0 / self.rate_hz
            self._last_t = ts
            true_g = self.platter_g + self.object_g
            k = 1.0 - math.exp(-dt / self.tau_s)
            self._shown_g += (true_g - self._shown_g) * k
            sigma = self.noise_g + (self.motor_noise_g if motor_moving else 0) + (self.fan_noise_g if fan_on else 0)
            g = self._shown_g + self.rng.gauss(0.0, sigma)
            out.append(int(round(self.zero_raw + self.counts_per_g * g)))
            self._next_sample_t += 1.0 / self.rate_hz
        return out


@dataclass
class ChannelState:
    name: str
    kind: str
    level: float = 0.0
    on_since: float = 0.0
    limit_s: float = 0.0                 # 0 = unlimited
    cool_until: float = 0.0
    held: bool = False                   # fan forced off while weighing
