"""Fingerspelling: per-frame letter classifier, moving letters and letter segmentation.

Pieces (all numpy, no heavy imports):

* ``LetterClassifier`` - small MLP over `features.letter_features` (weights in
  ``letters_<lang>.npz`` + ``.json``; the same network is exported to ONNX for reference,
  and a test checks numpy == ONNX Runtime == PyTorch).
* ``MotionTracker`` - the moving letters. ASL: J (I-hand, pinky draws a J) and Z (index
  draws a Z). LSP (MINEDU guide pp. 69-70, see lsp/GUIA_LSP.md): J, Z and N-tilde (N-hand
  moved side to side). A static-shape classifier cannot see movement, so we look at the
  trajectory of the relevant fingertip over the last ~1 s: path length (in palm units),
  sideways oscillations and direction changes. `temporal.py` holds the learned version.
* ``LetterSegmenter`` - decides WHEN a letter is written: the same letter must be the smoothed
  top-1 for `hold_s` while the hand is still; after a commit it re-arms when the hand moves
  (the little bounce used for double letters: "LL", "RR", "SS"), when another letter shows,
  or when the hand disappears.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import features as F
from . import modelstore

# Motion letters per sign language: letter -> (base static letters that look like it, finger tip index)
MOTION_LETTERS = {
    "ase": {"j": (("i", "j"), 20), "z": (("z", "d", "x", "g"), 8)},
    "prl": {"j": (("i", "j"), 20), "z": (("z", "d", "x", "g"), 8), "ñ": (("n",), 0)},
    "ils": {},
}
CONTROL_LABELS = ("SPACE", "BACK")


# ---------------------------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------------------------
class LetterClassifier:
    def __init__(self, weights: dict, meta: dict):
        self.meta = meta
        self.classes = list(meta["classes"])
        self.mu = np.asarray(weights["mu"], np.float32)
        self.sd = np.asarray(weights["sd"], np.float32)
        self.layers = []
        i = 0
        while "W%d" % i in weights:
            self.layers.append((np.asarray(weights["W%d" % i], np.float32), np.asarray(weights["b%d" % i], np.float32)))
            i += 1
        self.prototypes = {k: np.asarray(v, np.float32) for k, v in meta.get("prototypes", {}).items()}

    @classmethod
    def load(cls, lang: str, path: Optional[Path] = None) -> "LetterClassifier":
        p = Path(path) if path else modelstore.find_model("letters_%s.npz" % lang)
        if p is None or not Path(p).exists():
            raise FileNotFoundError("no letter model for %s (train with training/sign/train_letters.py)" % lang)
        meta = json.loads(Path(p).with_suffix(".json").read_text())
        with np.load(p) as z:
            weights = {k: z[k] for k in z.files}
        return cls(weights, meta)

    def logits(self, feats: np.ndarray) -> np.ndarray:
        x = (np.asarray(feats, np.float32) - self.mu) / self.sd
        # np.errstate: numpy 2.x + Apple Accelerate raise spurious "divide by zero in matmul"
        # warnings on finite inputs; the outputs are checked finite in the tests.
        with np.errstate(all="ignore"):
            for i, (W, b) in enumerate(self.layers):
                x = x @ W.T + b
                if i < len(self.layers) - 1:
                    x = np.maximum(x, 0.0)
        return x

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        z = self.logits(np.atleast_2d(feats))
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_hand(self, hand_xy: np.ndarray, is_left: bool) -> np.ndarray:
        q = F.canonical_hand(hand_xy, is_left)
        return self.predict_proba(F.letter_features(q)[None])[0]


# ---------------------------------------------------------------------------------------------
# Moving letters
# ---------------------------------------------------------------------------------------------
@dataclass
class MotionTracker:
    """Keeps ~1.2 s of the signing hand and tells whether J, Z or N-tilde was just drawn.

    A moving letter needs ALL of: (1) the static handshape stayed the letter's base shape for
    most of the window (so finger changes between letters do not count), (2) the WHOLE hand
    travelled (wrist path), (3) the tip path has the right form (Z: two sharp turns; N-tilde:
    side-to-side reversals). Positions are in palm units and smoothed (5-frame mean) before
    path lengths are measured, so keypoint jitter does not add up to fake motion."""
    lang: str
    window_s: float = 1.2
    j_path: float = 1.5          # pinky-tip path (palm units)
    j_wrist: float = 0.4
    z_path: float = 2.4          # index-tip path
    z_wrist: float = 1.2
    z_turns: int = 2
    enie_path: float = 1.2       # wrist path for the N-tilde side-to-side
    enie_reversals: int = 1
    hist: deque = field(default_factory=lambda: deque(maxlen=120))

    def reset(self) -> None:
        self.hist.clear()

    def push(self, t: float, hand_xy: np.ndarray, is_left: bool, static_top: str = "") -> None:
        p = np.asarray(hand_xy, np.float32).copy()
        if is_left:
            p[:, 0] = -p[:, 0]
        palm = float(np.mean(np.linalg.norm(p[[5, 9, 13, 17]] - p[0], axis=1)))
        self.hist.append((t, p, max(palm, 1e-6), static_top))

    def _window(self, now: float) -> list:
        return [h for h in self.hist if now - h[0] <= self.window_s]

    @staticmethod
    def _smooth(tr: np.ndarray, k: int = 5) -> np.ndarray:
        if len(tr) < k:
            return tr
        ker = np.ones(k) / k
        return np.stack([np.convolve(tr[:, d], ker, mode="valid") for d in range(tr.shape[1])], axis=1)

    def _track(self, win: list, idx: int) -> np.ndarray:
        palm = float(np.median([h[2] for h in win]))
        return self._smooth(np.array([h[1][idx] / palm for h in win], np.float32))

    @staticmethod
    def _path(tr: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum()) if len(tr) > 1 else 0.0

    @staticmethod
    def _rdp(pts: np.ndarray, eps: float) -> np.ndarray:
        """Ramer-Douglas-Peucker polyline simplification (keeps the corners of a drawn Z)."""
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        ab = b - a
        n = float(np.linalg.norm(ab))
        if n < 1e-9:
            d = np.linalg.norm(pts - a, axis=1)
        else:
            d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
        i = int(np.argmax(d))
        if d[i] <= eps:
            return np.stack([a, b])
        left = MotionTracker._rdp(pts[: i + 1], eps)
        right = MotionTracker._rdp(pts[i:], eps)
        return np.concatenate([left[:-1], right])

    @staticmethod
    def _turns(tr: np.ndarray, eps: float = 0.3, angle_deg: float = 100.0) -> int:
        """Sharp corners of the simplified path (a Z has 2, a straight swipe 0)."""
        if len(tr) < 3:
            return 0
        poly = MotionTracker._rdp(np.asarray(tr, np.float64), eps)
        if len(poly) < 3:
            return 0
        v = np.diff(poly, axis=0)
        cos_thr = math.cos(math.radians(angle_deg))
        n = 0
        for a, b in zip(v[:-1], v[1:]):
            if float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) < cos_thr:
                n += 1
        return n

    @staticmethod
    def _x_reversals(tr: np.ndarray, min_amp: float = 0.25) -> int:
        if len(tr) < 2:
            return 0
        x = tr[:, 0]
        rev, direction, anchor = 0, 0, x[0]
        for v in x[1:]:
            d = v - anchor
            if abs(d) >= min_amp:
                sgn = 1 if d > 0 else -1
                if direction and sgn != direction:
                    rev += 1
                direction = sgn
                anchor = v
        return rev

    def motion_amount(self, now: Optional[float] = None) -> float:
        if not self.hist:
            return 0.0
        win = self._window(self.hist[-1][0] if now is None else now)
        return self._path(self._track(win, 0)) if len(win) >= 5 else 0.0

    def detect(self, static_top: str = "", now: Optional[float] = None) -> Optional[str]:
        """Return 'j', 'z' or 'ñ' when that moving letter was just drawn, else None."""
        spec = MOTION_LETTERS.get(self.lang, {})
        if not spec or not self.hist:
            return None
        now = self.hist[-1][0] if now is None else now
        win = self._window(now)
        if len(win) < 8:
            return None
        for letter, (bases, tip) in spec.items():
            # the trailing run of frames with the letter's base handshape (gaps of <= 2 frames
            # allowed): the movement must happen WHILE the hand has that shape
            run, gap = [], 0
            for h in reversed(win):
                if h[3] in bases:
                    run.append(h)
                    gap = 0
                elif gap < 2 and run:
                    gap += 1
                else:
                    break
            run.reverse()
            if len(run) < 8:
                continue
            wrist = self._track(run, 0)
            wpath = self._path(wrist)
            tr = self._track(run, tip)
            if letter == "j" and self._path(tr) >= self.j_path and wpath >= self.j_wrist:
                return "j"
            if (letter == "z" and self._path(tr) >= self.z_path and wpath >= self.z_wrist
                    and self._turns(tr) >= self.z_turns):
                return "z"
            if letter == "ñ" and wpath >= self.enie_path and self._x_reversals(wrist) >= self.enie_reversals:
                return "ñ"
        return None


# ---------------------------------------------------------------------------------------------
# Segmentation: when is a letter "written"?
# ---------------------------------------------------------------------------------------------
@dataclass
class Commit:
    letter: str
    confidence: float
    alternatives: list
    t_start: float
    t_end: float


@dataclass
class LetterSegmenter:
    """Hold/transition logic, independent of the camera so it is unit-tested.

    Feed one frame at a time with ``update(t, probs, classes, still)``:
    probs = per-frame class probabilities (or None when no hand), still = the hand is not
    moving (palm units per second below `still_speed`)."""
    hold_s: float = 0.35
    min_conf: float = 0.55
    ema_tau_s: float = 0.12
    release_conf: float = 0.35
    rearm_motion: float = 0.35   # palm units the wrist moves away from a written letter = a new take
    _ema: Optional[np.ndarray] = None
    _last_t: Optional[float] = None
    _cand: Optional[str] = None
    _cand_since: float = 0.0
    _committed: Optional[str] = None
    _armed: bool = True
    _travel: float = 0.0
    _other_since: Optional[float] = None
    _commit_pos: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._ema = None
        self._last_t = None
        self._cand = None
        self._committed = None
        self._armed = True
        self._travel = 0.0
        self._other_since = None
        self._commit_pos = None

    def progress(self, now: float) -> float:
        if self._cand is None or not self._armed and self._cand == self._committed:
            return 0.0
        return float(min(1.0, (now - self._cand_since) / max(self.hold_s, 1e-6)))

    def smoothed(self) -> Optional[np.ndarray]:
        return None if self._ema is None else self._ema.copy()

    def update(self, t: float, probs: Optional[np.ndarray], classes: list, still: bool = True,
               wrist: Optional[np.ndarray] = None) -> Optional[Commit]:
        """wrist: position of the wrist in palm units (any origin). After a commit, moving it
        `rearm_motion` away from where the letter was written re-arms the same letter (double
        letters: sign L, bounce the hand a little, sign L again)."""
        if probs is None:                      # hand gone: everything re-arms
            self.reset()
            return None
        probs = np.asarray(probs, np.float32)
        if self._ema is None or self._last_t is None:
            self._ema = probs.copy()
        else:
            dt = max(t - self._last_t, 1e-3)
            a = 1.0 - math.exp(-dt / self.ema_tau_s)
            self._ema = (1 - a) * self._ema + a * probs
        self._last_t = t
        k = int(np.argmax(self._ema))
        top, conf = classes[k], float(self._ema[k])

        # re-arming after a commit
        if self._committed is not None and not self._armed:
            if wrist is not None and self._commit_pos is not None:
                self._travel = max(self._travel, float(np.linalg.norm(np.asarray(wrist) - self._commit_pos)))
            if self._travel >= self.rearm_motion or conf < self.release_conf:
                self._armed = True
            if top != self._committed and conf >= self.min_conf:
                if self._other_since is None:
                    self._other_since = t
                if t - self._other_since >= 0.12:
                    self._armed = True
            else:
                self._other_since = None

        if top != self._cand:
            self._cand = top
            self._cand_since = t
        if not still:
            self._cand_since = t               # the hold only counts while the hand is still
        held = t - self._cand_since
        if held >= self.hold_s and conf >= self.min_conf and self._armed:
            order = np.argsort(-self._ema)
            alts = [[classes[i], float(self._ema[i])] for i in order[1:4]]
            self._committed = top
            self._armed = False
            self._travel = 0.0
            self._commit_pos = None if wrist is None else np.asarray(wrist, np.float32).copy()
            self._other_since = None
            c = Commit(top, conf, alts, self._cand_since, t)
            self._cand_since = t
            return c
        return None

    def force_commit(self, letter: str, conf: float, t: float, alts: Optional[list] = None,
                     wrist: Optional[np.ndarray] = None) -> Commit:
        """Used for moving letters (their end is the end of the movement, not a hold)."""
        self._committed = letter
        self._armed = False
        self._travel = 0.0
        self._commit_pos = None if wrist is None else np.asarray(wrist, np.float32).copy()
        self._cand = letter
        self._cand_since = t
        return Commit(letter, conf, alts or [], t, t)
