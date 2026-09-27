"""Frame and keypoint sources for the SignEngine, plus the skeleton overlay.

* ``open_camera()``    real USB camera (V4L2 + MJPG on the Jetson, AVFoundation on a Mac) with a
                       grabber thread that always keeps only the NEWEST frame (no lag).
* ``SyntheticSource``  mock camera: synthetic fingerspelling/word keypoints, no model needed.
* ``ReplaySource``     replays keypoint recordings made with `python -m yq.sign.record`.
Keypoint sources return ready keypoints, so the engine skips the pose model for them.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import settings
from . import skeleton as sk
from . import synth


class LatestFrameCamera:
    """Wraps cv2.VideoCapture; a thread reads continuously, read() returns the newest frame."""

    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._t = 0.0
        self._n = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True, name="sign-cam")
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if not ok or f is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._t = f, time.monotonic()
                self._n += 1

    def read(self, last_n: int = -1, timeout: float = 1.0):
        """-> (ok, frame, t, n); waits for a frame newer than `last_n`."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._frame is not None and self._n != last_n:
                    return True, self._frame, self._t, self._n
            time.sleep(0.002)
        return False, None, 0.0, last_n

    def release(self):
        self._stop.set()
        self._th.join(timeout=1.0)
        self.cap.release()


def open_camera(spec: Optional[str] = None) -> LatestFrameCamera:
    import cv2
    spec = str(spec if spec is not None else settings.SIGN_CAMERA)
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(spec if spec.startswith("/dev/") else int(spec), cv2.CAP_V4L2)
        # order matters for V4L2: FOURCC first, then size and fps, then the 1-frame buffer
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.SIGN_CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.SIGN_CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, settings.SIGN_CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif sys.platform == "darwin":
        cap = cv2.VideoCapture(int(spec), cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.SIGN_CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.SIGN_CAM_HEIGHT)
    else:
        cap = cv2.VideoCapture(int(spec))
    if not cap.isOpened():
        raise RuntimeError("sign camera %s did not open (permission? cable?)" % spec)
    return LatestFrameCamera(cap)


class KeypointSource:
    """Base: next() -> (t, xy (69,2), conf (69,)) or None at the end."""
    is_keypoints = True
    image_wh = (synth.W, synth.H)

    def next(self):
        raise NotImplementedError

    def release(self):
        pass


class SyntheticSource(KeypointSource):
    """Loops a synthetic story: fingerspells `words` (letters mode) with the hand dropping
    between words. Real-time paced unless realtime=False (tests)."""

    def __init__(self, words=("condor", "sol"), shapes: Optional[dict] = None, fps: float = 30.0,
                 realtime: bool = True, loop: bool = True, seed: int = 0):
        self.words, self.shapes, self.fps = list(words), shapes or {}, fps
        self.realtime, self.loop = realtime, loop
        self.rng = np.random.default_rng(seed)
        self._gen = self._frames()
        self._t0 = time.monotonic()

    def _frames(self):
        t_off = 0.0
        while True:
            for w in self.words:
                last = 0.0
                for t, xy, cf in synth.letter_sequence(list(w), self.shapes, fps=self.fps, rng=self.rng):
                    last = t
                    yield t_off + t, xy, cf
                t_off += last + 1.0 / self.fps
                rest_xy, rest_cf = synth.body()
                for k in range(int(settings.SIGN_AUTO_SPACE_S * self.fps) + 10):
                    yield t_off, rest_xy + self.rng.normal(0, 1.0, rest_xy.shape).astype(np.float32), rest_cf
                    t_off += 1.0 / self.fps
            if not self.loop:
                return

    def next(self):
        try:
            t, xy, cf = next(self._gen)
        except StopIteration:
            return None
        if self.realtime:
            wait = self._t0 + t - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        return t, xy, cf


class ReplaySource(KeypointSource):
    """Replays recordings (.npz from yq.sign.record: xy (T,69,2), conf (T,69), t (T,))."""

    def __init__(self, paths, realtime: bool = True, loop: bool = False):
        self.paths = [Path(p) for p in paths]
        self.realtime, self.loop = realtime, loop
        self._gen = self._frames()
        self._t0 = time.monotonic()

    def _frames(self):
        off = 0.0
        while True:
            for p in self.paths:
                with np.load(p, allow_pickle=False) as z:
                    xy, cf, t = z["xy"], z["conf"], z["t"]
                    if "image_wh" in z.files:
                        self.image_wh = tuple(int(v) for v in z["image_wh"])
                for i in range(len(t)):
                    yield off + float(t[i] - t[0]), xy[i], cf[i]
                off += float(t[-1] - t[0]) + 1.0
            if not self.loop:
                return

    next = SyntheticSource.next


# ------------------------------------------------------------------------------------------
# Drawing (on the SAME frame the keypoints came from)
# ------------------------------------------------------------------------------------------
_EDGES = sk.canon_edges()


def draw_skeleton(img: np.ndarray, xy: np.ndarray, conf: np.ndarray, thr: float = 0.3,
                  highlight: Optional[str] = None) -> np.ndarray:
    import cv2
    for a, b in _EDGES:
        if conf[a] >= thr and conf[b] >= thr:
            in_hand = a >= sk.C_LHAND.start
            col = (80, 220, 255) if in_hand else (200, 200, 200)
            cv2.line(img, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), col, 2, cv2.LINE_AA)
    for i in range(sk.N_CANON):
        if conf[i] >= thr:
            side = "left" if sk.C_LHAND.start <= i < sk.C_LHAND.stop else ("right" if i >= sk.C_RHAND.start else None)
            col = (0, 200, 0) if side and side == highlight else ((0, 140, 255) if side else (255, 255, 255))
            cv2.circle(img, (int(xy[i, 0]), int(xy[i, 1])), 3 if side else 4, col, -1, cv2.LINE_AA)
    return img


def blank_canvas(wh: tuple) -> np.ndarray:
    img = np.zeros((wh[1], wh[0], 3), np.uint8)
    img[:] = (40, 32, 28)
    return img
