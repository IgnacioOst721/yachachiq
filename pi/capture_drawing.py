# capture_drawing.py -- takes a photo of the finished physical drawing
# Uses the same USB camera as the ASL system (or the Pi camera).
# Called by listen.py after each scene is plotted, or run standalone:
#   python3 capture_drawing.py stories/2026-08-30_15-00-00/scene_1_photo.png

import cv2
import sys
import time

CAMERA_INDEX = 0        # 0 = first camera; change if the wrong camera opens
WARMUP_FRAMES = 15      # let the camera adjust exposure before shooting


def capture(out_path, camera_index=CAMERA_INDEX):
    """Take one photo and save it to out_path. Returns True on success."""
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        print(f"  ⚠️  No se pudo abrir la cámara {camera_index}")
        return False
    try:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # warm up: discard the first frames so exposure settles
        for _ in range(WARMUP_FRAMES):
            cam.read()
            time.sleep(0.05)
        ok, frame = cam.read()
        if not ok:
            print("  ⚠️  La cámara no devolvió imagen")
            return False
        cv2.imwrite(out_path, frame)
        print(f"  📷 Foto del dibujo guardada: {out_path}")
        return True
    finally:
        cam.release()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "drawing_photo.png"
    input("Coloca el dibujo frente a la cámara y presiona ENTER...")
    capture(out)
