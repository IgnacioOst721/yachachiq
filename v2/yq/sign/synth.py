"""Synthetic keypoint sequences (no camera, no model): for the mock engine, tests and for
exercising the training pipeline. They are NOT used to claim any accuracy.

Coordinates: canonical 69-point layout (skeleton.py), pixels of a 1280x800 non-mirrored
camera image; the person faces the camera, so their RIGHT side is on the image LEFT.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import skeleton as sk

W, H = 1280, 800
# canonical right hand (wrist-centred, palm = 1, y down): a relaxed open hand
OPEN_HAND = np.array([
    [0, 0], [-0.35, -0.25], [-0.6, -0.5], [-0.8, -0.75], [-0.95, -1.0],
    [-0.3, -1.0], [-0.35, -1.45], [-0.38, -1.75], [-0.4, -2.0],
    [0.0, -1.05], [0.0, -1.55], [0.0, -1.9], [0.0, -2.15],
    [0.28, -0.98], [0.32, -1.45], [0.35, -1.75], [0.37, -2.0],
    [0.52, -0.85], [0.6, -1.2], [0.65, -1.45], [0.7, -1.65]], np.float32)


def body(cx: float = 640, shoulder_w: float = 260, top: float = 330) -> tuple:
    """A person at rest (hands down), canonical xy (69,2) and conf (69,)."""
    xy = np.zeros((sk.N_CANON, 2), np.float32)
    s = shoulder_w
    ys = top
    xy[sk.C_NOSE] = [cx, ys - 0.62 * s]
    xy[1] = [cx + 0.13 * s, ys - 0.72 * s]      # left eye (person's left = image right)
    xy[2] = [cx - 0.13 * s, ys - 0.72 * s]
    xy[3] = [cx + 0.3 * s, ys - 0.66 * s]
    xy[4] = [cx - 0.3 * s, ys - 0.66 * s]
    xy[sk.C_LSH] = [cx + 0.5 * s, ys]
    xy[sk.C_RSH] = [cx - 0.5 * s, ys]
    xy[sk.C_LEL] = [cx + 0.62 * s, ys + 0.8 * s]
    xy[sk.C_REL] = [cx - 0.62 * s, ys + 0.8 * s]
    xy[sk.C_LWR] = [cx + 0.6 * s, ys + 1.5 * s]
    xy[sk.C_RWR] = [cx - 0.6 * s, ys + 1.5 * s]
    xy[sk.C_LHIP] = [cx + 0.35 * s, ys + 1.7 * s]
    xy[sk.C_RHIP] = [cx - 0.35 * s, ys + 1.7 * s]
    # face subset around the nose
    f = sk.C_FACE.start
    face = [(-0.22, -0.2), (-0.14, -0.23), (-0.05, -0.21), (0.05, -0.21), (0.14, -0.23), (0.22, -0.2),
            (0.0, -0.12), (0.0, 0.0), (-0.1, 0.12), (0.0, 0.1), (0.1, 0.12), (0.0, 0.16), (0.0, 0.12), (0.0, 0.14)]
    for i, (dx, dy) in enumerate(face):
        xy[f + i] = xy[sk.C_NOSE] + np.array([-dx * s, dy * s])   # person's left = image right
    conf = np.full(sk.N_CANON, 0.9, np.float32)
    place_hand(xy, conf, "right", OPEN_HAND, xy[sk.C_RWR], 0.28 * s)
    place_hand(xy, conf, "left", OPEN_HAND, xy[sk.C_LWR], 0.28 * s)
    return xy, conf


def place_hand(xy: np.ndarray, conf: np.ndarray, side: str, shape: np.ndarray, wrist: np.ndarray,
               palm_px: float, visible: bool = True) -> None:
    """Put a canonical right-hand shape at `wrist` (the left hand is the mirror image)."""
    q = np.asarray(shape, np.float32).copy()
    if side == "left":
        q[:, 0] = -q[:, 0]
    sl = sk.C_LHAND if side == "left" else sk.C_RHAND
    xy[sl] = wrist[None, :] + q * palm_px
    conf[sl] = 0.9 if visible else 0.05
    wr = sk.C_LWR if side == "left" else sk.C_RWR
    xy[wr] = wrist


def to_wholebody(xy: np.ndarray, conf: np.ndarray) -> tuple:
    k = np.zeros((sk.N_WHOLEBODY, 2), np.float32)
    s = np.zeros(sk.N_WHOLEBODY, np.float32)
    idx = np.asarray(sk.CANON_WHOLEBODY)
    k[idx] = xy
    s[idx] = conf
    return k, s


def _motion_offset(letter: str, u: float, palm: float) -> np.ndarray:
    """Displacement of the hand during a moving letter, u in 0..1 (image pixels)."""
    if letter == "j":        # hook: down, then curve toward the body and up
        a = np.pi * u
        return np.array([-1.2 * palm * (1 - np.cos(a)) / 2, 1.4 * palm * np.sin(a)])
    if letter == "z":        # right, diagonal down-left, right (drawn by the index)
        pts = np.array([[0, 0], [1.4, 0], [0, 1.2], [1.4, 1.2]]) * palm
        seg = min(int(u * 3), 2)
        f = u * 3 - seg
        return pts[seg] + (pts[seg + 1] - pts[seg]) * f
    if letter == "ñ":        # side to side
        return np.array([0.9 * palm * np.sin(2 * np.pi * 1.5 * u), 0.0])
    return np.zeros(2)


def letter_sequence(letters: list, shapes: dict, fps: float = 30.0, hold_s: float = 0.7,
                    move_s: float = 0.25, motion_s: float = 0.8, noise_px: float = 0.8,
                    rng: Optional[np.random.Generator] = None, drop_between: tuple = ()):
    """Right hand fingerspelling `letters` near the right shoulder. shapes: letter -> canonical
    hand (21,2). Motion letters j/z/ñ use the shape of their base letter plus a movement.
    drop_between: indices i where the hand briefly drops between letter i-1 and i.
    Yields (t, xy, conf)."""
    rng = rng or np.random.default_rng(0)
    xy0, cf0 = body()
    s = float(np.linalg.norm(xy0[sk.C_LSH] - xy0[sk.C_RSH]))
    palm = 0.28 * s
    home = xy0[sk.C_RSH] + np.array([-0.1 * s, -0.05 * s])
    base = {"j": "i", "ñ": "n"}
    t = 0.0
    prev_shape = OPEN_HAND
    prev_pos = home.copy()
    dt = 1.0 / fps
    for i, letter in enumerate(letters):
        shp = shapes.get(letter, shapes.get(base.get(letter, letter), OPEN_HAND))
        if i in drop_between:
            for k in range(int(0.3 * fps)):
                xy, cf = xy0.copy(), cf0.copy()
                place_hand(xy, cf, "right", OPEN_HAND, xy0[sk.C_RWR], palm)
                yield t, xy + rng.normal(0, noise_px, xy.shape).astype(np.float32), cf
                t += dt
        n_move = max(1, int(move_s * fps))
        for k in range(n_move):                   # transition: blend shapes
            f = (k + 1) / n_move
            xy, cf = xy0.copy(), cf0.copy()
            # a repeated letter (LL, RR) is signed twice with a small bounce/slide in between
            bob = 0.6 if (i > 0 and letters[i - 1] == letter) else 0.3
            pos = (1 - f) * prev_pos + f * home + np.array([0, bob * palm * np.sin(np.pi * f)])
            place_hand(xy, cf, "right", (1 - f) * prev_shape + f * shp, pos, palm)
            yield t, xy + rng.normal(0, noise_px, xy.shape).astype(np.float32), cf
            t += dt
        dur = motion_s if letter in ("j", "z", "ñ") else hold_s
        n = max(1, int(dur * fps))
        for k in range(n):
            xy, cf = xy0.copy(), cf0.copy()
            off = _motion_offset(letter, k / max(n - 1, 1), palm) if letter in ("j", "z", "ñ") else 0.0
            place_hand(xy, cf, "right", shp, home + off, palm)
            prev_pos = home + off
            yield t, xy + rng.normal(0, noise_px, xy.shape).astype(np.float32), cf
            t += dt
        prev_shape = shp


def word_sequence(cls: int, n_classes: int, rng: np.random.Generator, fps: float = 30.0):
    """A synthetic 'sign' for class `cls`: both hands follow a class-specific trajectory
    (location, path shape, repetitions). Returns (xy (T,69,2), conf (T,69))."""
    xy0, cf0 = body(cx=640 + rng.normal(0, 40), shoulder_w=260 * rng.uniform(0.85, 1.15))
    s = float(np.linalg.norm(xy0[sk.C_LSH] - xy0[sk.C_RSH]))
    palm = 0.28 * s
    crng = np.random.default_rng(1000 + cls)        # fixed per class
    center = xy0[sk.C_RSH] + np.array([crng.uniform(0.2, 1.0), crng.uniform(-0.8, 0.6)]) * s
    radius = crng.uniform(0.15, 0.45) * s
    reps = int(crng.integers(1, 4))
    kind = int(crng.integers(0, 3))
    two = bool(crng.integers(0, 2))
    shape = OPEN_HAND * np.array([1.0, crng.uniform(0.5, 1.0)], np.float32)
    dur = rng.uniform(0.8, 1.5)
    T = int(dur * fps)
    xs, cs = [], []
    for k in range(T):
        u = k / max(T - 1, 1)
        a = 2 * np.pi * reps * u
        if kind == 0:
            off = np.array([np.cos(a), np.sin(a)]) * radius
        elif kind == 1:
            off = np.array([np.sin(a), 0.0]) * radius
        else:
            off = np.array([0.0, np.sin(a)]) * radius
        off = off + rng.normal(0, 0.03 * s, 2)
        xy, cf = xy0.copy(), cf0.copy()
        place_hand(xy, cf, "right", shape, center + off, palm)
        if two:
            mirror_c = np.array([2 * xy0[sk.C_NOSE, 0] - center[0], center[1]])
            place_hand(xy, cf, "left", shape, mirror_c + off * np.array([-1, 1]), palm)
        xs.append(xy + rng.normal(0, 1.5, xy.shape))
        cs.append(cf)
    return np.stack(xs).astype(np.float32), np.stack(cs).astype(np.float32)


def two_handed(cls: int) -> bool:
    """Whether synthetic word class `cls` uses both hands (replays word_sequence's draws)."""
    crng = np.random.default_rng(1000 + cls)
    crng.uniform(0.2, 1.0)
    crng.uniform(-0.8, 0.6)
    crng.uniform(0.15, 0.45)
    crng.integers(1, 4)
    crng.integers(0, 3)
    return bool(crng.integers(0, 2))
