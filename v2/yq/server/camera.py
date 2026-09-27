"""Visitor-facing camera for the consent portrait (like v1).

The rule shown on screen: covering the camera at any moment = "no, don't publish".
Two consecutive covered readings (~1 s) are needed so a hand passing by does not
count. Buttons on the screen always win. Face/smile detection only helps pick the
nicest portrait frame and is optional (OpenCV 5 wheels no longer ship the Haar
cascade files; JetPack's OpenCV 4 does).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from yq.common import config
from yq.server import mock_assets, settings

log = logging.getLogger("yq.server.camera")


class MockCamera:
    mode = "mock:ui"

    def __init__(self):
        self.covered = False               # tests / gear menu "simulate covered camera"

    def grab_jpeg(self) -> Optional[bytes]:
        return mock_assets.portrait_frame(time.time(), self.covered)

    def close(self) -> None:
        pass


class PortraitCamera:
    mode = "opencv"

    def __init__(self, device: Optional[str] = None):
        self.device = str(device if device is not None else settings.PORTRAIT_CAMERA)
        self._cap = None
        self._lock = threading.Lock()
        self.covered = False               # unused (API parity with MockCamera)

    def _open(self):
        import cv2
        dev = int(self.device) if self.device.isdigit() else self.device
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if str(dev).startswith("/dev/") else cv2.VideoCapture(dev)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def grab_jpeg(self) -> Optional[bytes]:
        with self._lock:
            try:
                import cv2
                if self._cap is None:
                    self._cap = self._open()
                if self._cap is None:
                    return None
                ok, frame = self._cap.read()
                if not ok:
                    self._cap.release()
                    self._cap = None
                    return None
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                return jpg.tobytes() if ok else None
            except Exception as e:
                log.warning("portrait camera failed: %s", e)
                return None

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None


def make_camera():
    if settings.ui_mocked("camera") or config.mock("sign_camera") or config.mock("cameras"):
        return MockCamera()
    return PortraitCamera()


_cascades: dict = {}


def _cascade(name: str):
    if name not in _cascades:
        import os
        import cv2
        path = os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""), name)
        _cascades[name] = cv2.CascadeClassifier(path) if os.path.exists(path) else None
    return _cascades[name]


def look_at(jpeg: Optional[bytes]) -> Optional[dict]:
    """jpeg -> {bright, covered, face, smile}; face/smile are None when no detector is installed."""
    if not jpeg:
        return None
    try:
        import cv2
        import numpy as np
        arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    if arr is None:
        return None
    bright, std = float(arr.mean()), float(arr.std())
    out = {"bright": round(bright, 1),
           "covered": bright < float(settings.COVERED_BRIGHTNESS) or std < float(settings.COVERED_STD),
           "face": None, "smile": None}
    if not out["covered"]:
        try:
            face_c = _cascade("haarcascade_frontalface_default.xml")
            if face_c is not None:
                faces = face_c.detectMultiScale(arr, 1.2, 5, minSize=(60, 60))
                out["face"] = bool(len(faces))
                smile_c = _cascade("haarcascade_smile.xml")
                if len(faces) and smile_c is not None:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    roi = arr[y + h // 2:y + h, x:x + w]
                    out["smile"] = len(smile_c.detectMultiScale(roi, 1.7, 22, minSize=(25, 25))) > 0
        except Exception:
            pass
    return out


def decide(looks: list, vote: Optional[bool]) -> tuple:
    """(publish, reason). Button vote wins; two covered readings in a row = no; no camera = no."""
    if vote is not None:
        return bool(vote), ("botón: sí" if vote else "botón: no")
    if not looks:
        return False, "sin cámara para preguntar"
    for a, b in zip(looks, looks[1:]):
        if a["covered"] and b["covered"]:
            return False, "cámara tapada"
    if len(looks) == 1 and looks[0]["covered"]:
        return False, "cámara tapada"
    return True, "cámara abierta" + (" · sonrisa" if any(x.get("smile") for x in looks[-6:]) else "")


def score(look: dict) -> float:
    return look["bright"] + (200 if look.get("face") else 0) + (100 if look.get("smile") else 0)
