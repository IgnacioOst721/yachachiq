"""Closed-loop protection of the object during active thermography.

While the halogen is on, every thermal frame is compared with the baseline
(median of the frames recorded before heating). The watched region is the
object: pixels inside a central ROI that warmed more than `warm_px_c`; while
too few pixels have warmed yet, the whole ROI is watched. Pixels that were
already hot in the baseline (e.g. the lamp fixture after a previous scan) are
ignored. The halogen must go off as soon as either

    delta_t = P99(frame - baseline) over the region  >= max_dt_c   ("delta_t")
    temp    = P99(frame)            over the region  >= max_abs_c  ("abs_limit")

or the camera stops delivering fresh frames for `max_frozen_s` ("camera_frozen":
we cannot see, so we stop). The planned heat time ("time") and the firmware
cap (45 s, not changed here) remain as outer limits.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import settings as S


class HeatGuard:
    def __init__(self, max_dt_c: float = None, max_abs_c: float = None, roi=None, warm_px_c: float = None,
                 min_warm_px: int = None, percentile: float = None, max_frozen_s: float = None):
        self.max_dt_c = float(S.THERMAL_MAX_DT_C if max_dt_c is None else max_dt_c)
        self.max_abs_c = float(S.THERMAL_MAX_ABS_C if max_abs_c is None else max_abs_c)
        self.roi = tuple(S.THERMAL_GUARD_ROI if roi is None else roi)          # x0, x1, y0, y1 fractions
        self.warm_px_c = float(S.THERMAL_WARM_PX_C if warm_px_c is None else warm_px_c)
        self.min_warm_px = int(S.THERMAL_MIN_WARM_PX if min_warm_px is None else min_warm_px)
        self.percentile = float(S.THERMAL_GUARD_PERCENTILE if percentile is None else percentile)
        self.max_frozen_s = float(S.THERMAL_MAX_FROZEN_S if max_frozen_s is None else max_frozen_s)
        self._base_frames: list = []
        self.baseline: Optional[np.ndarray] = None
        self.region: Optional[np.ndarray] = None
        self.excluded_hot = 0
        self.max_dt = 0.0
        self.max_temp = None
        self.max_dt_heating = 0.0
        self.last = {}
        self._prev = None
        self._frozen_since = None

    # -- baseline ---------------------------------------------------------------------------------
    def add_baseline(self, frame_c) -> None:
        self._base_frames.append(np.asarray(frame_c, dtype=np.float32))

    def start(self) -> None:
        if not self._base_frames:
            raise RuntimeError("no baseline thermal frames")
        self.baseline = np.median(np.stack(self._base_frames), axis=0)
        h, w = self.baseline.shape
        x0, x1, y0, y1 = self.roi
        roi = np.zeros((h, w), bool)
        roi[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = True
        hot = self.baseline >= self.max_abs_c - 1.0
        self.excluded_hot = int((roi & hot).sum())
        self.region = roi & ~hot
        if self.region.sum() == 0:
            raise RuntimeError("thermal guard: no usable pixels in the ROI")
        self.max_temp = float(np.percentile(self.baseline[self.region], self.percentile))

    # -- per frame ----------------------------------------------------------------------------------
    def _measure(self, frame_c):
        frame = np.asarray(frame_c, dtype=np.float32)
        rise = frame - self.baseline
        warmed = self.region & (rise > self.warm_px_c)
        use = warmed if warmed.sum() >= self.min_warm_px else self.region
        dt = float(np.percentile(rise[use], self.percentile))
        temp = float(np.percentile(frame[use], self.percentile))
        self.max_dt = max(self.max_dt, dt)
        self.max_temp = max(self.max_temp, temp)
        self.last = {"dt_c": dt, "temp_c": temp, "warm_px": int(warmed.sum()),
                     "region": "warmed" if use is warmed else "roi"}
        return dt, temp

    def update(self, t: float, frame_c) -> Optional[str]:
        """Call for every frame while the halogen is on. Returns a stop reason or None."""
        frame = np.asarray(frame_c)
        if self._prev is not None and np.array_equal(frame, self._prev):
            if self._frozen_since is None:
                self._frozen_since = t
            if t - self._frozen_since >= self.max_frozen_s:
                return "camera_frozen"
            return None                  # same picture again: nothing new to judge
        self._frozen_since = None
        self._prev = frame.copy()
        dt, temp = self._measure(frame)
        self.max_dt_heating = max(self.max_dt_heating, dt)
        if dt >= self.max_dt_c:
            return "delta_t"
        if temp >= self.max_abs_c:
            return "abs_limit"
        return None

    def observe(self, t: float, frame_c) -> None:
        """Cooling phase: only track the maxima (surface can still rise a bit)."""
        self._measure(frame_c)

    def summary(self) -> dict:
        return {"max_dt_c": round(self.max_dt, 3), "max_temp_c": round(self.max_temp, 3) if self.max_temp else None,
                "limits": {"max_dt_c": self.max_dt_c, "max_abs_c": self.max_abs_c},
                "guard": {"roi": list(self.roi), "percentile": self.percentile, "warm_px_c": self.warm_px_c,
                          "min_warm_px": self.min_warm_px, "excluded_hot_px": self.excluded_hot,
                          "max_frozen_s": self.max_frozen_s}}
