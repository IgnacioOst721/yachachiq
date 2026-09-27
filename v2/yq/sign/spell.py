"""Fingerspelling state machine: frames of keypoints in, letters and words out.

Speller.step(t, xy, conf) per frame (canonical 69 keypoints). It picks the signing hand,
classifies the handshape, detects the moving letters, commits letters with the hold logic
and builds the current word with lexicon completions. Camera-independent and unit-tested.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from . import features as F
from . import lexicon as lx
from . import settings
from .letters import CONTROL_LABELS, LetterClassifier, LetterSegmenter, MotionTracker

MOTION_ONLY = ("j", "z", "ñ")      # never committed by a hold, only by their movement


class Speller:
    def __init__(self, sign_lang: str, clf: LetterClassifier, lex: Optional[lx.Lexicon] = None,
                 on_letter: Optional[Callable] = None, on_word_end: Optional[Callable] = None):
        self.lang = sign_lang
        self.clf = clf
        self.lex = lex
        self.on_letter = on_letter          # (letter, conf, alts, t0, t1)
        self.on_word_end = on_word_end      # () -> None, the hand rested after a word
        self.classes = list(clf.classes)
        self.static_classes = [c for c in self.classes if c not in MOTION_ONLY]
        self._static_idx = [self.classes.index(c) for c in self.static_classes]
        self.seg = LetterSegmenter(hold_s=settings.SIGN_HOLD_S, min_conf=settings.SIGN_LETTER_MIN_CONF)
        self.motion = MotionTracker(sign_lang)
        self.side: Optional[str] = None
        self.letters: list = []             # [{letter: prob}] of the current word
        self.hand_gone_since: Optional[float] = None
        self.current = ("", 0.0)            # smoothed top letter now
        self.progress = 0.0
        self._prev_wrist = None
        self._speed = 0.0
        self._palm: Optional[float] = None
        self._anchor = None
        self._motion_armed = True

    # -- word buffer ------------------------------------------------------------------
    def buffer(self) -> str:
        return "".join(max(o, key=o.get) for o in self.letters)

    def candidates(self, n: int = 5, final: bool = False) -> list:
        if not self.letters:
            return []
        if self.lex is None:
            return [(self.buffer(), 1.0)]
        return lx.complete(self.lex, self.letters, n=n, final=final)

    def backspace_letter(self) -> bool:
        if self.letters:
            self.letters.pop()
            return True
        return False

    def clear(self) -> None:
        self.letters = []
        self.seg.reset()
        self.motion.reset()

    # -- per frame -----------------------------------------------------------------------
    def _static_probs(self, probs: np.ndarray) -> np.ndarray:
        p = probs[self._static_idx].copy()
        if "j" in self.classes and "i" in self.static_classes:          # a J frame looks like an I
            p[self.static_classes.index("i")] += probs[self.classes.index("j")]
        s = p.sum()
        return p / s if s > 0 else p

    def step(self, t: float, xy: np.ndarray, conf: np.ndarray) -> Optional[tuple]:
        """Returns (letter, conf, alternatives) when a letter was written this frame."""
        side = F.pick_signing_hand(xy, conf, self.side)
        if side is None:
            self.side = None
            self.seg.update(t, None, self.static_classes)
            self.motion.reset()
            self._prev_wrist = None
            self._palm = None
            self._anchor = None
            self._motion_armed = True
            self.current, self.progress = ("", 0.0), 0.0
            if self.hand_gone_since is None:
                self.hand_gone_since = t
            elif self.letters and t - self.hand_gone_since >= settings.SIGN_AUTO_SPACE_S:
                self.hand_gone_since = t + 1e9          # fire once
                if self.on_word_end:
                    self.on_word_end()
            return None
        self.hand_gone_since = None
        if side != self.side:
            self.seg.reset()
            self.motion.reset()
            self._prev_wrist = None
        self.side = side
        hand, _hc = F.hand_block(xy, conf, side)
        is_left = side == "left"
        probs = self.clf.predict_hand(hand, is_left)
        palm_now = float(np.mean(np.linalg.norm(hand[[5, 9, 13, 17]] - hand[0], axis=1))) or 1.0
        self._palm = palm_now if self._palm is None else 0.9 * self._palm + 0.1 * palm_now
        # pixel differences divided by a smoothed palm size: a handshape change (which changes
        # the 2D palm size) must not look like movement
        if self._prev_wrist is not None:
            dt = max(t - self._prev_wrist[0], 1e-3)
            step = float(np.linalg.norm(hand[0] - self._prev_wrist[1])) / self._palm
            self._speed = 0.6 * self._speed + 0.4 * (step / dt)
        self._prev_wrist = (t, hand[0].copy())
        if self._anchor is None:
            self._anchor = hand[0].copy()
        wrist = (hand[0] - self._anchor) / self._palm     # palm units from where the hand appeared
        still = self._speed < settings.SIGN_STILL_SPEED
        raw_top = self.classes[int(np.argmax(probs))]
        self.motion.push(t, hand, is_left, raw_top)
        moved = self.motion.detect(raw_top, t) if self._motion_armed else None
        if moved:
            base = {"j": "i", "ñ": "n"}.get(moved, moved)
            c = float(probs[self.classes.index(base)]) if base in self.classes else 0.5
            alts = [[{"j": "i", "ñ": "n"}.get(moved, "x"), 0.2]]
            self.seg.force_commit(moved, c, t, alts, wrist)
            self.motion.reset()
            self._motion_armed = False          # until a held letter or the hand leaves
            return self._written(moved, max(c, 0.5), alts, t, t)

        sp = self._static_probs(probs)
        commit = self.seg.update(t, sp, self.static_classes, still=still, wrist=wrist)
        if commit is not None:
            self._motion_armed = True
        ema = self.seg.smoothed()
        if ema is not None:
            k = int(np.argmax(ema))
            self.current = (self.static_classes[k], float(ema[k]))
        self.progress = self.seg.progress(t)
        if commit is None:
            return None
        return self._written(commit.letter, commit.confidence, commit.alternatives, commit.t_start, commit.t_end)

    def _written(self, letter: str, c: float, alts: list, t0: float, t1: float):
        if letter not in CONTROL_LABELS:
            obs = {letter: c}
            for a, p in alts:
                if a not in CONTROL_LABELS and a not in obs:
                    obs[a] = float(p)
            self.letters.append(obs)
        if self.on_letter:
            self.on_letter(letter, c, alts, t0, t1)
        return letter, c, alts
