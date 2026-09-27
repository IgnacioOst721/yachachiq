"""Whole-body pose for sign language: RTMW (133 keypoints) + person tracking + One Euro filter.

    est = PoseEstimator()                 # loads ONNX files from MODELS_DIR/sign/pose (offline)
    pf = est.process(frame_bgr, t)        # PoseFrame: filtered keypoints of the visitor
    pf.canon_xy, pf.canon_conf            # 69 canonical points (see skeleton.py)

Speed tricks: the person detector (YOLOX-tiny) only runs to (re)acquire the visitor or every
SIGN_REDETECT_S; otherwise the crop comes from the previous keypoints (single-person
tracking). For fingerspelling the signing hand can be refined with RTMPose-m hand
(256x256), which is more precise on fingers than the whole-body model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import features as F
from . import modelstore, settings
from . import skeleton as sk
from .filters import OneEuroFilter


@dataclass
class PoseFrame:
    t: float
    frame_id: int
    image_wh: tuple
    kpts: np.ndarray                  # (133, 2) filtered, pixels
    scores: np.ndarray                # (133,)
    raw: np.ndarray                   # (133, 2) unfiltered
    bbox: Optional[np.ndarray] = None
    person: bool = False
    timings_ms: dict = field(default_factory=dict)
    canon_xy: np.ndarray = None       # (69, 2)
    canon_conf: np.ndarray = None     # (69,)

    def __post_init__(self):
        if self.canon_xy is None:
            self.canon_xy, self.canon_conf = F.from_wholebody(self.kpts, self.scores)


def empty_frame(t: float, frame_id: int, wh: tuple) -> PoseFrame:
    z = np.zeros((sk.N_WHOLEBODY, 2), np.float32)
    return PoseFrame(t, frame_id, wh, z, np.zeros(sk.N_WHOLEBODY, np.float32), z.copy())


def _box_from_kpts(kpts: np.ndarray, scores: np.ndarray, wh: tuple, thr: float) -> Optional[np.ndarray]:
    good = scores[:sk.LHAND0 + 42] >= thr
    good[13:23] = False                      # ignore legs/feet: visitors are framed from the waist up
    if good.sum() < 6:
        return None
    p = kpts[:sk.LHAND0 + 42][good]
    x1, y1 = p.min(0)
    x2, y2 = p.max(0)
    w, h = x2 - x1, y2 - y1
    side = max(w, h, 0.15 * wh[1])
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w2, h2 = max(w, side * 0.75) * 1.15 / 2, max(h, side) * 1.15 / 2
    return np.array([max(0, cx - w2), max(0, cy - h2), min(wh[0], cx + w2), min(wh[1], cy + h2)], np.float32)


def _hand_box(hand_xy: np.ndarray, hand_conf: np.ndarray, thr: float) -> Optional[np.ndarray]:
    good = hand_conf >= thr
    if good.sum() < 8:
        return None
    p = hand_xy[good]
    c = (p.min(0) + p.max(0)) / 2
    side = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))) * 1.35
    side = max(side, 40.0)
    return np.array([c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2], np.float32)


class PoseEstimator:
    def __init__(self, model: Optional[str] = None, backend: Optional[str] = None,
                 detector: Optional[str] = None, hand_refine: Optional[str] = None):
        from . import rtm
        self.model_name = model or settings.SIGN_POSE_MODEL
        self.backend = backend or settings.SIGN_BACKEND
        det_name = settings.SIGN_DETECTOR if detector is None else detector
        self.hand_refine = settings.SIGN_HAND_REFINE if hand_refine is None else hand_refine
        self.thr = settings.SIGN_KPT_THR
        cache = modelstore.sign_dir() / "trt_cache"
        path = modelstore.pose_model_path(self.model_name)
        if path is None:
            raise FileNotFoundError("pose model %s missing: run `python -m yq.sign.modelstore download`"
                                    % self.model_name)
        wh = modelstore.POSE_MODELS[self.model_name][1]
        self.pose = rtm.TopDownPose(path, wh, self.backend, cache)
        self.detector = None
        if det_name:
            dp = modelstore.pose_model_path(det_name)
            if dp is not None:
                self.detector = rtm.PersonDetector(dp, modelstore.POSE_MODELS[det_name][1], self.backend, cache)
        self.hand_model = None
        if self.hand_refine != "off":
            hp = modelstore.pose_model_path("rtmpose-hand")
            if hp is not None:
                self.hand_model = rtm.TopDownPose(hp, (256, 256), self.backend, cache)
        self.providers = self.pose.providers
        self.euro = OneEuroFilter(settings.SIGN_EURO_MIN_CUTOFF, settings.SIGN_EURO_BETA,
                                  settings.SIGN_EURO_D_CUTOFF, min_conf=self.thr)
        self._track: Optional[np.ndarray] = None
        self._last_det = -1e9
        self._frame_id = 0
        self.refine_side: Optional[str] = None     # set by the engine in letters mode

    def reset(self) -> None:
        self._track = None
        self.euro.reset()

    def _detect(self, img: np.ndarray, wh: tuple) -> Optional[np.ndarray]:
        if self.detector is None:
            return np.array([0, 0, wh[0], wh[1]], np.float32)
        boxes = self.detector(img)
        if len(boxes) == 0:
            return None
        cx = wh[0] / 2
        # the visitor = big and central
        score = [(b[2] - b[0]) * (b[3] - b[1]) * b[4] / (1.0 + abs((b[0] + b[2]) / 2 - cx) / wh[0]) for b in boxes]
        return boxes[int(np.argmax(score))][:4].astype(np.float32)

    def process(self, img_bgr: np.ndarray, t: Optional[float] = None) -> PoseFrame:
        t = time.monotonic() if t is None else t
        self._frame_id += 1
        h, w = img_bgr.shape[:2]
        wh = (w, h)
        tm = {}
        t0 = time.perf_counter()
        box = self._track
        if box is None or (t - self._last_det) >= settings.SIGN_REDETECT_S:
            det = self._detect(img_bgr, wh)
            self._last_det = t
            tm["det"] = (time.perf_counter() - t0) * 1e3
            if det is not None:
                box = det if box is None else det            # fresh box wins
        if box is None:
            self.euro.reset()
            return empty_frame(t, self._frame_id, wh)
        t1 = time.perf_counter()
        kp, sc = self.pose(img_bgr, box)
        tm["pose"] = (time.perf_counter() - t1) * 1e3
        upper = sc[[0, 5, 6]]
        if float(np.mean(upper)) < self.thr:                  # lost the person
            self._track = None
            self.euro.reset()
            return PoseFrame(t, self._frame_id, wh, kp, sc, kp.copy(), box, False, tm)
        if self.hand_model is not None and self.refine_side in ("left", "right"):
            t2 = time.perf_counter()
            self._refine_hand(img_bgr, kp, sc, self.refine_side)
            tm["hand"] = (time.perf_counter() - t2) * 1e3
        raw = kp.copy()
        scale = float(box[3] - box[1])
        filt = self.euro(kp, t, sc, scale=scale)
        self._track = _box_from_kpts(filt, sc, wh, self.thr)
        tm["total"] = (time.perf_counter() - t0) * 1e3
        return PoseFrame(t, self._frame_id, wh, filt, sc, raw, box, True, tm)

    def _refine_hand(self, img: np.ndarray, kp: np.ndarray, sc: np.ndarray, side: str) -> None:
        base = sk.LHAND0 if side == "left" else sk.RHAND0
        hb = _hand_box(kp[base:base + 21], sc[base:base + 21], self.thr)
        if hb is None:
            return
        hk, hs = self.hand_model(img, hb)
        # keep the refined point where the hand model is at least as confident
        better = hs >= sc[base:base + 21] * 0.9
        kp[base:base + 21][better] = hk[better]
        sc[base:base + 21] = np.maximum(sc[base:base + 21], hs)
