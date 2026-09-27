"""Isolated word signs: cut the stream into signs, classify each with the skeleton transformer.

* ``SignSpotter`` finds where one sign starts and ends from hand activity: a sign is "on"
  while at least one hand is raised (above the rest line between shoulders and hips) or
  moving; it ends after `rest_s` of hands down/still, or at `max_s`.
* ``WordRecognizer`` runs ``words_<lang>.onnx`` (exported by training/sign/train_words.py) on
  the normalised, resampled segment and returns the top-k glosses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import features as F
from . import modelstore
from . import skeleton as sk


class WordRecognizer:
    def __init__(self, onnx_path: Path, meta: dict, backend: str = "cpu"):
        from .rtm import make_session
        self.meta = meta
        self.classes = list(meta["classes"])
        self.frames = int(meta["frames"])
        self.text = meta.get("text", {})          # gloss -> spoken-language word
        self.sess = make_session(onnx_path, backend)
        self.inp = self.sess.get_inputs()[0].name

    @classmethod
    def load(cls, lang: str, backend: str = "cpu", path: Optional[Path] = None) -> "WordRecognizer":
        p = Path(path) if path else modelstore.find_model("words_%s.onnx" % lang)
        if p is None or not Path(p).exists():
            raise FileNotFoundError("no word model for %s (record signs, then training/sign/train_words.py)" % lang)
        meta = json.loads(Path(p).with_suffix(".json").read_text())
        return cls(Path(p), meta, backend)

    def prepare(self, seq_xy: np.ndarray, seq_conf: np.ndarray) -> np.ndarray:
        feats, _ = F.normalize_sequence(seq_xy, seq_conf)
        return F.resample_sequence(feats, self.frames)[None]

    def predict(self, seq_xy: np.ndarray, seq_conf: np.ndarray, k: int = 5) -> list:
        """-> [(gloss, prob), ...] best first."""
        x = self.prepare(seq_xy, seq_conf)
        logits = self.sess.run(None, {self.inp: x})[0][0]
        z = np.exp(logits - logits.max())
        p = z / z.sum()
        order = np.argsort(-p)[:k]
        return [(self.classes[i], float(p[i])) for i in order]


@dataclass
class SignSpotter:
    rest_s: float = 0.45
    max_s: float = 4.0
    min_s: float = 0.35
    move_thr: float = 0.9          # shoulder widths per second = "moving"
    thr: float = 0.3
    _buf_xy: list = field(default_factory=list)
    _buf_cf: list = field(default_factory=list)
    _buf_t: list = field(default_factory=list)
    _active: bool = False
    _quiet_since: Optional[float] = None
    _prev: Optional[tuple] = None

    def reset(self) -> None:
        self._buf_xy, self._buf_cf, self._buf_t = [], [], []
        self._active = False
        self._quiet_since = None
        self._prev = None

    @property
    def active(self) -> bool:
        return self._active

    def _hands_up_or_moving(self, t: float, xy: np.ndarray, cf: np.ndarray) -> bool:
        s = F.body_scale(xy, cf, self.thr)
        if not s:
            return False
        up = False
        speed = 0.0
        hip_ok = cf[sk.C_LHIP] >= self.thr and cf[sk.C_RHIP] >= self.thr
        sh_y = (xy[sk.C_LSH, 1] + xy[sk.C_RSH, 1]) / 2
        rest_y = ((xy[sk.C_LHIP, 1] + xy[sk.C_RHIP, 1]) / 2 + sh_y) / 2 if hip_ok else sh_y + 1.3 * s
        wr = []
        for w, h0 in ((sk.C_LWR, sk.C_LHAND.start), (sk.C_RWR, sk.C_RHAND.start)):
            if cf[h0] >= self.thr or cf[w] >= self.thr:
                p = xy[h0] if cf[h0] >= self.thr else xy[w]
                wr.append(p)
                if p[1] < rest_y:
                    up = True
            else:
                wr.append(None)
        if self._prev is not None:
            pt, pw = self._prev
            dt = max(t - pt, 1e-3)
            for a, b in zip(wr, pw):
                if a is not None and b is not None:
                    speed = max(speed, float(np.linalg.norm(a - b)) / s / dt)
        self._prev = (t, wr)
        return up or speed > self.move_thr

    def update(self, t: float, xy: np.ndarray, cf: np.ndarray):
        """Feed one frame (canonical xy, conf). Returns (seq_xy, seq_conf, t0, t1) when a sign ended."""
        busy = self._hands_up_or_moving(t, xy, cf)
        if busy:
            self._quiet_since = None
            if not self._active:
                self._active = True
                self._buf_xy, self._buf_cf, self._buf_t = [], [], []
        elif self._active and self._quiet_since is None:
            self._quiet_since = t
        if self._active:
            self._buf_xy.append(xy.copy())
            self._buf_cf.append(cf.copy())
            self._buf_t.append(t)
            dur = t - self._buf_t[0]
            ended = (self._quiet_since is not None and t - self._quiet_since >= self.rest_s) or dur >= self.max_s
            if ended:
                self._active = False
                self._quiet_since = None
                if dur >= self.min_s:
                    # drop the trailing rest frames
                    keep = [i for i, tt in enumerate(self._buf_t) if tt <= t - (self.rest_s if dur < self.max_s else 0)]
                    keep = keep or list(range(len(self._buf_t)))
                    out = (np.stack([self._buf_xy[i] for i in keep]), np.stack([self._buf_cf[i] for i in keep]),
                           self._buf_t[0], self._buf_t[keep[-1]])
                    self._buf_xy, self._buf_cf, self._buf_t = [], [], []
                    return out
        return None
