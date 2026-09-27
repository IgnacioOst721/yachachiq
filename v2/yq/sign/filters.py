"""One Euro filter for keypoints (Casiez, Roussel, Vogel, CHI 2012; https://gery.casiez.net/1euro/).

Low-pass filter whose cutoff rises with speed: still hands get strong smoothing
(no jitter, which is what made v1's dots flicker), fast hands get little lag.

    alpha(cutoff, dt) = 1 / (1 + tau/dt),  tau = 1 / (2*pi*cutoff)
    dx_hat = lowpass(dx, alpha(d_cutoff))
    cutoff = min_cutoff + beta * |dx_hat|
    x_hat  = lowpass(x, alpha(cutoff))

Vectorised over any number of keypoints. Points with low confidence are not
used to update the filter (they keep their last filtered value, and after
`hold_s` without a good reading the filter resets for that point).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _alpha(cutoff, dt: float):
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff, dtype=np.float64))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0,
                 min_conf: float = 0.3, hold_s: float = 0.25):
        """min_cutoff in Hz; beta in 1/(units of x); coordinates are in the units you feed
        (we feed pixels / body scale so beta is tuned for that; see settings)."""
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.min_conf = float(min_conf)
        self.hold_s = float(hold_s)
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: Optional[float] = None
        self._last_good: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray, t: float, conf: Optional[np.ndarray] = None,
                 scale: float = 1.0) -> np.ndarray:
        """x: (N, D) array, t: seconds, conf: (N,) or None. Returns filtered (N, D).

        `scale` (e.g. the person's box height in pixels) divides the speed before `beta`, like
        MediaPipe's landmark smoothing, so the same beta works near and far from the camera."""
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        good = np.ones(n, dtype=bool) if conf is None else (np.asarray(conf) >= self.min_conf)
        if self._x is None or self._x.shape != x.shape or self._t is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            self._last_good = np.where(good, t, -1e9)
            return self._x.astype(np.float32)
        dt = t - self._t
        if dt <= 0:
            dt = 1e-3
        self._t = t
        # points that were lost for too long restart from the new reading
        stale = (t - self._last_good) > self.hold_s
        restart = good & stale
        if restart.any():
            self._x[restart] = x[restart]
            self._dx[restart] = 0.0
        upd = good & ~stale
        if upd.any():
            dx = (x[upd] - self._x[upd]) / dt
            a_d = _alpha(self.d_cutoff, dt)
            dx_hat = a_d * dx + (1.0 - a_d) * self._dx[upd]
            speed = (np.linalg.norm(dx_hat, axis=-1, keepdims=True) if dx_hat.ndim > 1 else np.abs(dx_hat)) / max(scale, 1e-6)
            cutoff = self.min_cutoff + self.beta * speed
            a = _alpha(cutoff, dt)
            self._x[upd] = a * x[upd] + (1.0 - a) * self._x[upd]
            self._dx[upd] = dx_hat
        self._last_good = np.where(good, t, self._last_good)
        return self._x.astype(np.float32)
