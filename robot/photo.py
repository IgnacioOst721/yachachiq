"""Photograph the finished drawing with the camera mounted over the plotter bed.

capture(out_path) -> True/False.  Camera chosen by config.PHOTO_CAMERA:
    ""                       first camera
    "1"                      OpenCV index
    "/dev/v4l/by-id/..."     a fixed device path (does not change when you reboot)
Falls back to rpicam-still / libcamera-still (Pi Camera Module). In mock mode
it writes the pen-preview PNG instead so the rest of the flow can be tested.
"""
import logging
import os
import shutil
import subprocess
import sys
import time

import config

log = logging.getLogger("photo")


def mode():
    if not config.PHOTO_ENABLED:
        return "off"
    if config.MOCK:
        return "mock"
    return "camera"


def _open(cam):
    import cv2
    if isinstance(cam, str) and cam.startswith("/dev/"):
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(0)
    else:
        idx = int(cam) if str(cam).strip().isdigit() else 0
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(idx)
    return cap


def _capture_cv2(out_path):
    import cv2
    cap = _open(config.PHOTO_CAMERA)
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(15):                     # let exposure settle
            cap.read()
            time.sleep(0.04)
        ok, frame = cap.read()
        if not ok:
            return False
        cv2.imwrite(str(out_path), frame)
        return True
    finally:
        cap.release()


def _capture_picam(out_path):
    for cmd in ("rpicam-still", "libcamera-still"):
        if shutil.which(cmd):
            r = subprocess.run([cmd, "-o", str(out_path), "-t", "1500", "--width", "1280", "--height", "720", "-n"],
                               capture_output=True, timeout=25)
            if r.returncode == 0:
                return True
    return False


def capture(out_path, fallback_png=None):
    if mode() == "off":
        return False
    if mode() == "mock":
        if fallback_png and os.path.exists(fallback_png):
            shutil.copy(fallback_png, out_path)
            return True
        return False
    time.sleep(float(config.PHOTO_SETTLE_SECONDS))
    try:
        if _capture_cv2(out_path):
            log.info("photo saved: %s", out_path)
            return True
    except Exception as e:
        log.warning("cv2 capture failed: %s", e)
    if _capture_picam(out_path):
        log.info("photo saved (picam): %s", out_path)
        return True
    log.warning("no camera available for the drawing photo")
    return False


def list_cameras():
    """Stable device paths on Linux (use these in PHOTO_CAMERA / LSP_CAMERA)."""
    d = "/dev/v4l/by-id"
    if os.path.isdir(d):
        return sorted(os.path.join(d, f) for f in os.listdir(d) if "index0" in f)
    return []
