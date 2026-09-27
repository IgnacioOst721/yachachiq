"""Box cameras A (20 deg) and B (55 deg): Arducam IMX519 16 MP USB 3.0 UVC.

Rules (docs/box.md): focus, exposure, gain and white balance are LOCKED for a
whole scan (uvcctl.lock_camera); only one camera streams at a time (USB
bandwidth): open, capture the pass, close; stale buffered frames are flushed
after every turntable move; every still is checked for sharpness/exposure and
retried. In mock mode (mock("cameras")) images are rendered by mockcam.
"""
from __future__ import annotations

import glob
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

from yq.common import config

from . import settings as S
from . import uvcctl

log = logging.getLogger("yq.box.cameras")


# -- discovery -------------------------------------------------------------------------------------
def _sysfs_name(dev: str) -> str:
    base = os.path.basename(os.path.realpath(dev))
    try:
        return Path("/sys/class/video4linux/%s/name" % base).read_text().strip()
    except OSError:
        return ""


def list_capture_devices(hints) -> list:
    """Capture nodes (video-index0) whose by-id link or card name matches a hint.
    by-path links are preferred: two identical cameras can share one by-id name."""
    out, seen = [], set()
    for pattern in ("/dev/v4l/by-path/*-video-index0", "/dev/v4l/by-id/*-video-index0"):
        for p in sorted(glob.glob(pattern)):
            real = os.path.realpath(p)
            label = os.path.basename(p) + " " + _sysfs_name(p)
            if real in seen or not any(h.lower() in label.lower() for h in hints):
                continue
            seen.add(real)
            out.append(p)
    return out


def find_box_cameras() -> dict:
    """{"A": device, "B": device}. YQ_BOX_CAMERA_A/_B win; otherwise the two
    matching devices sorted by USB port path (A = first). Set the env vars once
    the cameras are in their final ports (docs/box.md, "Cámaras")."""
    found = {"A": S.CAMERA_A_DEVICE or None, "B": S.CAMERA_B_DEVICE or None}
    if all(found.values()):
        return found
    devs = [d for d in list_capture_devices(S.CAMERA_NAME_HINTS) if d not in found.values()]
    for name in ("A", "B"):
        if not found[name] and devs:
            found[name] = devs.pop(0)
    missing = [k for k, v in found.items() if not v]
    if missing:
        raise RuntimeError("no encuentro la(s) cámara(s) %s (conectadas por USB 3.0?)" % ", ".join(missing))
    return found


# -- image checks -------------------------------------------------------------------------------------
def image_stats(img) -> dict:
    """Sharpness (variance of the Laplacian on a 1000 px copy, inside the lit
    region), mean brightness and fraction of clipped pixels."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    s = min(1.0, 1000.0 / max(h, w))    # never upscale (it would hide sharpness)
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(small, cv2.CV_64F)
    lit = cv2.dilate((small > 25).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    sharp = float(lap[lit].var()) if lit.sum() > 200 else float(lap.var())
    return {"sharpness": round(sharp, 2), "mean": round(float(small.mean()), 2),
            "clipped": round(float((small >= 250).mean()), 5), "lit_fraction": round(float(lit.mean()), 4)}


def check_image(stats: dict, expect: str, min_sharpness: float) -> list:
    """Problems (Spanish) for an image that should be `expect`: normal | rti | uv | dark."""
    probs = []
    if expect == "dark":
        if stats["mean"] > 40:
            probs.append("la imagen oscura no está oscura (¿entra luz por la puerta?)")
        return probs
    if expect in ("normal", "rti") and stats["lit_fraction"] < 0.002:
        probs.append("imagen casi negra (¿luz apagada o cámara tapada?)")
    if expect == "normal" and stats["clipped"] > 0.02:
        probs.append("imagen sobreexpuesta (%.1f %% saturado)" % (100 * stats["clipped"]))
    if expect == "normal" and stats["sharpness"] < min_sharpness:
        probs.append("imagen borrosa (nitidez %.1f < %.1f)" % (stats["sharpness"], min_sharpness))
    return probs


# -- real UVC camera -------------------------------------------------------------------------------------
class UvcCamera:
    def __init__(self, name: str, device: str, resolution=None, fourcc: str = None):
        self.name = name
        self.device = device
        self.resolution = tuple(resolution or S.CAMERA_RESOLUTION)
        self.fourcc = fourcc or S.CAMERA_FOURCC
        self.controls = dict(S.CAMERA_CONTROLS.get(name, S.CAMERA_CONTROLS["A"]))
        self.cap = None

    @property
    def is_open(self) -> bool:
        return self.cap is not None

    def open(self) -> "UvcCamera":
        import cv2
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("no puedo abrir la cámara %s (%s)" % (self.name, self.device))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if got != self.resolution:
            log.warning("camera %s: asked %s, got %s", self.name, self.resolution, got)
            self.resolution = got
        self.cap = cap
        return self

    def lock(self, focus=None, exposure_us=None, gain=None, wb_k=None) -> dict:
        c = self.controls
        res = uvcctl.lock_camera(self.device, focus=c["focus"] if focus is None else focus,
                                 exposure_us=c["exposure_us"] if exposure_us is None else exposure_us,
                                 gain=c["gain"] if gain is None else gain, wb_k=c["wb_k"] if wb_k is None else wb_k)
        for k, v in res["applied"].items():
            self.controls[{"exposure": "exposure_us", "wb": "wb_k"}.get(k, k)] = v
        return res

    def warm_up(self, frames: int = None) -> None:
        for _ in range(S.CAMERA_WARMUP_FRAMES if frames is None else frames):
            self.cap.grab()

    def read(self):
        if self.cap is None:
            raise RuntimeError("camera %s not open" % self.name)
        for _ in range(S.CAMERA_FLUSH_FRAMES):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("la cámara %s no entregó imagen" % self.name)
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_camera(name: str, scene_fn: Optional[Callable[[], dict]] = None, resolution=None):
    """Opened camera A or B (real, or MockCamera when mock("cameras")), controls locked."""
    if config.mock("cameras"):
        from .mockcam import MockCamera
        if scene_fn is None:
            from .device import get_box
            box = get_box()
            scene_fn = box.sim.scene if box.sim is not None else (lambda: {"platter_deg": box.platter_deg,
                                                                            "lights": {}, "object_g": 800.0})
        cam = MockCamera(name, scene_fn, resolution).open()
    else:
        cam = UvcCamera(name, find_box_cameras()[name], resolution).open()
    try:
        cam.lock_info = cam.lock()          # focus / exposure / gain / WB fixed for the whole pass
        cam.warm_up()                       # first frames after a control change are discarded
    except Exception:
        cam.close()
        raise
    return cam


def capture_still(cam, path, expect: str = "normal", min_sharpness: float = None, retries: int = None) -> dict:
    """Capture one still into `path` (JPEG), checking it; retries blurry/bad
    frames and keeps the best one. Returns {"file","stats","attempts","warnings"}."""
    import cv2
    min_sharpness = S.CAMERA_MIN_SHARPNESS if min_sharpness is None else min_sharpness
    retries = S.CAMERA_RETRIES if retries is None else retries
    best, best_stats, probs = None, None, []
    for attempt in range(retries + 1):
        frame = cam.read()
        stats = image_stats(frame)
        probs = check_image(stats, expect, min_sharpness)
        if best is None or stats["sharpness"] > best_stats["sharpness"]:
            best, best_stats = frame, stats
        if not probs:
            break
        time.sleep(0.2)   # let a vibration die down
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), best, [cv2.IMWRITE_JPEG_QUALITY, S.CAMERA_JPEG_QUALITY]):
        raise RuntimeError("no pude guardar %s" % path)
    return {"file": path.name, "stats": best_stats, "attempts": attempt + 1,
            "warnings": ["%s: %s" % (path.name, p) for p in probs]}
