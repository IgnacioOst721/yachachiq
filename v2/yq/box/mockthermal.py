"""Mock Lepton 3.5: physically plausible active thermography of a vessel with a
subsurface defect, seen by the thermal camera T (side view, 0 deg elevation).

Two-layer model per pixel of the object: a thin surface layer heated by the
halogen (when the simulated halogen channel is on), exchanging heat with the
bulk below it and with the air (Newton cooling), plus lateral diffusion. Over
the defect (a delamination / air gap) the surface-to-bulk conductance is low,
so that spot heats faster and cools later: the "hot spot" BOX-ANALYSIS must
find. Output frames are TLinear centikelvin uint16, like the real camera, with
Lepton-like noise (NETD ~50 mK) and one FFC freeze.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from . import settings as S
from .mockcam import PROFILE_R, PROFILE_Z
from .thermal import c_to_centikelvin

DEFECT = {"y_mm": 12.0, "z_mm": 62.0, "radius_mm": 9.0}


class MockThermal:
    fps = S.THERMAL_FPS

    def __init__(self, scene_fn: Callable[[], dict], ambient_c: float = 22.0, seed: int = 3,
                 ffc_at_s: tuple = (42.0,), ffc_len_s: float = 0.35):
        self.scene_fn = scene_fn
        self.ambient = ambient_c
        self.rng = np.random.default_rng(seed)
        self.ffc_at, self.ffc_len = ffc_at_s, ffc_len_s
        w, h = S.THERMAL_SIZE
        g = S.CAMERA_GEOMETRY["T"]
        dist = abs(g["position_mm"][0])                      # camera on the -X wall, looking at +X
        mm_per_px = 2 * dist * math.tan(math.radians(g["hfov_deg"] / 2)) / w
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        y = (u - w / 2) * mm_per_px
        z = g["aim_mm"][2] - (v - h / 2) * mm_per_px
        r = np.interp(np.clip(z, 0, PROFILE_Z[-1]), PROFILE_Z, PROFILE_R)
        self.mask = (z > 0) & (z < PROFILE_Z[-1]) & (np.abs(y) < r)
        self.platter = (~self.mask) & (z < 0) & (z > -6) & (np.abs(y) < S.PLATTER_RADIUS_MM)
        self.defect = self.mask & ((y - DEFECT["y_mm"]) ** 2 + (z - DEFECT["z_mm"]) ** 2 < DEFECT["radius_mm"] ** 2)
        # halogen above and to the right: more flux on top, cosine falloff on the curved sides
        cosang = np.sqrt(np.clip(1 - (y / np.maximum(r, 1)) ** 2, 0, 1))
        self.flux = np.where(self.mask, (0.55 + 0.45 * z / PROFILE_Z[-1]) * (0.4 + 0.6 * cosang), 0.0)
        self.g = np.where(self.defect, 0.035, 0.22)          # surface -> bulk conductance (1/s)
        self.q0 = 1.05                                        # C/s at full flux
        self.h_air = 0.012
        self.ts = np.full((h, w), ambient_c)                  # surface
        self.tb = np.full((h, w), ambient_c)                  # bulk
        self.mm_per_px = mm_per_px
        self.virtual_time = 0.0
        self._last = None
        self._t_real = None

    def open(self) -> "MockThermal":
        self._t_real = time.monotonic()
        return self

    def _step(self, dt: float, heating: bool) -> None:
        ts, tb = self.ts, self.tb
        lap = (np.roll(ts, 1, 0) + np.roll(ts, -1, 0) + np.roll(ts, 1, 1) + np.roll(ts, -1, 1) - 4 * ts)
        q = self.q0 * self.flux if heating else 0.0
        d_ts = q - self.g * (ts - tb) - self.h_air * (ts - self.ambient) + 0.6 * lap * self.mask
        d_tb = 0.08 * self.g * (ts - tb) - 0.002 * (tb - self.ambient)
        self.ts = np.where(self.mask, ts + dt * d_ts, self.ambient)
        self.tb = np.where(self.mask, tb + dt * d_tb, self.ambient)

    def read_u16(self) -> np.ndarray:
        scene = self.scene_fn()
        scale = float(scene.get("time_scale", S.SIM_TIME_SCALE) or 1.0)
        dt = 1.0 / self.fps
        # pace like the real camera (8.7 fps), faster when the simulator runs faster
        if self._t_real is not None:
            target = self._t_real + dt / scale
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._t_real = max(target, time.monotonic() - 1.0)
        heating = scene.get("lights", {}).get("halogen", 0.0) > 0 and scene.get("object_g", 0.0) > 1.0
        for _ in range(4):                                    # sub-steps for stability
            self._step(dt / 4, heating)
        self.virtual_time += dt
        frozen = any(a <= self.virtual_time < a + self.ffc_len for a in self.ffc_at)
        if frozen and self._last is not None:
            return self._last.copy()
        frame_c = self.ts + self.rng.normal(0, 0.04, self.ts.shape)
        self._last = c_to_centikelvin(frame_c)
        return self._last.copy()

    def close(self) -> None:
        self._t_real = None
