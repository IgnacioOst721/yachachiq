"""Thermal camera: GroupGets PureThermal 3 + FLIR Lepton 3.5 (160x120, radiometric).

The PureThermal firmware streams UVC "Y16" frames; with radiometry in TLinear
mode each pixel is the scene temperature in centikelvin (GroupGets
purethermal1-uvc-capture, uvc-radiometry.py: ktoc(v) = (v - 27315) / 100).
OpenCV's V4L2 backend returns Y16 as a 16-bit single-channel image when
CAP_PROP_CONVERT_RGB is 0 (opencv modules/videoio/src/cap_v4l.cpp: V4L2_PIX_FMT_Y16
-> CV_16UC1). The Lepton freezes for a moment during its flat-field correction
(FFC, shutter): consecutive identical frames are flagged as frozen.
"""
from __future__ import annotations

import glob
import os
import time
from typing import Callable, Optional

import numpy as np

from yq.common import config

from . import settings as S

TLINEAR_MIN, TLINEAR_MAX = 23315, 42315      # -40 .. +150 C in centikelvin


def centikelvin_to_c(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32) / 100.0 - 273.15


def c_to_centikelvin(c) -> np.ndarray:
    return np.clip(np.round((np.asarray(c, dtype=np.float64) + 273.15) * 100.0), 0, 65535).astype(np.uint16)


def to_frame_u16(raw) -> np.ndarray:
    """Normalize what the capture returned into a (120, 160) uint16 frame.
    Accepts a (120,160) uint16 image, a (122,160) frame with 2 telemetry rows
    (dropped), or a flat byte buffer of 160*120*2 bytes (little endian)."""
    a = np.asarray(raw)
    w, h = S.THERMAL_SIZE
    if a.dtype == np.uint8 and a.size == w * h * 2:
        a = a.reshape(-1).view("<u2").reshape(h, w)
    if a.dtype != np.uint16:
        raise ValueError("thermal frame is %s, expected uint16 Y16 (is CONVERT_RGB off?)" % a.dtype)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape == (h + 2, w):
        a = a[:h]
    if a.shape != (h, w):
        raise ValueError("thermal frame shape %s, expected (%d, %d)" % (a.shape, h, w))
    return a


def looks_like_tlinear(frame_u16: np.ndarray) -> bool:
    med = float(np.median(frame_u16))
    return TLINEAR_MIN <= med <= TLINEAR_MAX


def detect_frozen(frames_u16) -> np.ndarray:
    """Boolean per frame: identical to the previous one (FFC freeze or a stalled stream).
    Real sensor noise (~0.05 K = 5 counts) makes exact repeats otherwise impossible."""
    frames = np.asarray(frames_u16)
    frozen = np.zeros(len(frames), dtype=bool)
    if len(frames) > 1:
        same = np.all(frames[1:] == frames[:-1], axis=(1, 2))
        frozen[1:] = same
    return frozen


def find_thermal_device() -> str:
    if S.THERMAL_DEVICE:
        return S.THERMAL_DEVICE
    for pattern in ("/dev/v4l/by-id/*-video-index0", "/dev/v4l/by-path/*-video-index0"):
        for p in sorted(glob.glob(pattern)):
            if any(h.lower() in os.path.basename(p).lower() for h in S.THERMAL_NAME_HINTS):
                return p
    raise RuntimeError("no encuentro la cámara térmica PureThermal (¿cable USB?)")


class ThermalCamera:
    """Real PureThermal over V4L2."""

    fps = S.THERMAL_FPS

    def __init__(self, device: Optional[str] = None):
        self.device = device or find_thermal_device()
        self.cap = None

    def open(self) -> "ThermalCamera":
        import cv2
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("no puedo abrir la cámara térmica %s" % self.device)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, S.THERMAL_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, S.THERMAL_SIZE[1])
        self.cap = cap
        first = self.read_u16()
        if not looks_like_tlinear(first):
            self.close()
            raise RuntimeError("la térmica no está en modo radiométrico TLinear (mediana %d)" % np.median(first))
        return self

    def read_u16(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("la cámara térmica no entregó imagen")
        return to_frame_u16(frame)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_thermal(scene_fn: Optional[Callable[[], dict]] = None):
    if config.mock("thermal"):
        from .mockthermal import MockThermal
        if scene_fn is None:
            from .device import get_box
            box = get_box()
            scene_fn = box.sim.scene if box.sim is not None else (lambda: {"lights": {}, "object_g": 800.0})
        return MockThermal(scene_fn).open()
    return ThermalCamera().open()


def record(cam, duration_s: float, time_scale: float = 1.0, on_frame: Optional[Callable] = None,
           cancel_event=None) -> dict:
    """Record for `duration_s` (virtual seconds when time_scale > 1, mock only).
    on_frame(t_s, frame_c) is called after every frame (it may switch the
    halogen); if it returns a number, that becomes the new total duration (the
    cooling window restarts when heating stops early).
    Returns {"frames_c" (T,120,160) float32, "times" (T,), "frozen" (T,) bool}."""
    raws, times = [], []
    t0 = time.monotonic()
    while True:
        if cancel_event is not None and cancel_event.is_set():
            break
        raw = cam.read_u16()
        t = (time.monotonic() - t0) * time_scale
        if getattr(cam, "virtual_time", None) is not None:
            t = cam.virtual_time
        raws.append(raw)
        times.append(t)
        if on_frame is not None:
            new_duration = on_frame(t, centikelvin_to_c(raw))
            if new_duration is not None:
                duration_s = float(new_duration)
        if t >= duration_s:
            break
    arr = np.stack(raws) if raws else np.zeros((0,) + S.THERMAL_SIZE[::-1], np.uint16)
    return {"frames_c": centikelvin_to_c(arr), "times": np.asarray(times, dtype=np.float64),
            "frozen": detect_frozen(arr), "raw": arr}
