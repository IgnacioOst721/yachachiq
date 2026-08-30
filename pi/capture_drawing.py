# capture_drawing.py -- automatic photo of the finished drawing (runs on the Pi)
# A dedicated camera is mounted above the plotter bed. When the CNC finishes,
# listen.py calls capture() and the photo is taken automatically -- no human.
#
# Works with either:
#   - a USB webcam plugged into the Pi (uses OpenCV), or
#   - the official Raspberry Pi Camera Module (uses rpicam-still/libcamera-still)
#
# Standalone test:  python3 capture_drawing.py test.png

import subprocess
import sys
import time

CAMERA_INDEX = 0        # USB webcam index on the Pi
WARMUP_FRAMES = 15      # let the camera adjust exposure before shooting
SETTLE_SECONDS = 2      # wait after the plotter stops (vibration, pen lift)


def _capture_usb(out_path):
    import cv2
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        return False
    try:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(WARMUP_FRAMES):
            cam.read()
            time.sleep(0.05)
        ok, frame = cam.read()
        if not ok:
            return False
        cv2.imwrite(out_path, frame)
        return True
    finally:
        cam.release()


def _capture_picam(out_path):
    for cmd in ("rpicam-still", "libcamera-still"):
        try:
            r = subprocess.run([cmd, "-o", out_path, "-t", "1500",
                                "--width", "1280", "--height", "720", "-n"],
                               capture_output=True, timeout=20)
            if r.returncode == 0:
                return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False


def capture(out_path):
    """Automatic capture: settle, then try USB webcam, then Pi camera."""
    time.sleep(SETTLE_SECONDS)
    if _capture_usb(out_path) or _capture_picam(out_path):
        print(f"  📷 Foto del dibujo guardada: {out_path}")
        return True
    print("  ⚠️  Ninguna cámara disponible en la Pi (ni USB ni Pi Camera)")
    return False


if __name__ == "__main__":
    capture(sys.argv[1] if len(sys.argv) > 1 else "drawing_photo.png")
