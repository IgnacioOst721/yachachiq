"""From raw keypoints to the numbers the recognizers see.

Coordinates everywhere are ISOTROPIC image units (pixels, or MediaPipe's
normalised x multiplied by the image aspect W/H), y pointing down.

Two normalisations:

* letters (one hand): wrist-centred, divided by the palm size, turned into a
  canonical RIGHT hand (a left hand is mirrored), so the same handshape gives
  the same numbers wherever the hand is, however big it looks, and whichever
  hand signs. Rotation is kept on purpose (H vs U, G/Q, K/P differ only by
  orientation).
* words (whole upper body over time): shoulder-centred, divided by the shoulder
  width; left-dominant signers are mirrored so the model only sees
  right-dominant signing. Missing points are zeros with a 0 in the mask.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import skeleton as sk

HAND_MIN_CONF = 0.3       # RTMW score below this = point missing
HAND_MIN_POINTS = 15      # fewer good hand points than this = hand not usable


# --------------------------------------------------------------------------------------------
# Layout conversion
# --------------------------------------------------------------------------------------------
def from_wholebody(kpts: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """RTMW output (133, 2) + (133,) -> canonical (69, 2) + (69,)."""
    idx = np.asarray(sk.CANON_WHOLEBODY)
    return np.asarray(kpts, np.float32)[idx].copy(), np.asarray(scores, np.float32)[idx].copy()


def from_mp_holistic(pose: Optional[np.ndarray], face: Optional[np.ndarray], left_hand: Optional[np.ndarray],
                     right_hand: Optional[np.ndarray], aspect: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """MediaPipe Holistic landmarks (normalised x, y; NaN or None = missing) -> canonical.

    pose (33, >=2), face (468, >=2), hands (21, >=2). `aspect` = image W/H, multiplies x so
    distances are isotropic. The hands are assigned to the body side whose wrist is closer
    (so the data's left/right naming convention cannot silently swap them)."""
    xy = np.zeros((sk.N_CANON, 2), np.float32)
    ok = np.zeros(sk.N_CANON, np.float32)

    def put(dst: slice, src: Optional[np.ndarray], idx=None):
        if src is None:
            return
        a = np.asarray(src, np.float32)[:, :2]
        if idx is not None:
            a = a[np.asarray(idx)]
        good = np.isfinite(a).all(axis=1)
        a = np.where(good[:, None], a, 0.0)
        a[:, 0] *= aspect
        xy[dst] = a
        ok[dst] = good.astype(np.float32)

    put(sk.C_BODY, pose, sk.MP_POSE_TO_CANON_BODY)
    put(sk.C_FACE, face, sk.MP_FACE_TO_CANON_FACE)
    hands = [h for h in (left_hand, right_hand)]
    put(sk.C_LHAND, hands[0])
    put(sk.C_RHAND, hands[1])
    _fix_hand_sides(xy, ok)
    return xy, ok


def _fix_hand_sides(xy: np.ndarray, ok: np.ndarray) -> None:
    """Swap the two hand blocks if each hand's wrist is closer to the OTHER body wrist."""
    lw, rw = sk.C_LWR, sk.C_RWR
    if not (ok[lw] and ok[rw]):
        return
    l0, r0 = sk.C_LHAND.start, sk.C_RHAND.start
    have_l, have_r = ok[l0] > 0, ok[r0] > 0
    d = lambda a, b: float(np.linalg.norm(xy[a] - xy[b]))
    swap = False
    if have_l and have_r:
        swap = d(l0, lw) + d(r0, rw) > d(l0, rw) + d(r0, lw)
    elif have_l:
        swap = d(l0, lw) > d(l0, rw)
    elif have_r:
        swap = d(r0, rw) > d(r0, lw)
    if swap:
        for a in (xy, ok):
            tmp = a[sk.C_LHAND].copy()
            a[sk.C_LHAND] = a[sk.C_RHAND]
            a[sk.C_RHAND] = tmp


def mp_hand_to_xy(landmarks: np.ndarray, aspect: float) -> np.ndarray:
    """MediaPipe Hands normalised (21, >=2) -> isotropic (21, 2) in the COCO hand order."""
    a = np.asarray(landmarks, np.float32)[:, :2][np.asarray(sk.MP_HAND_TO_COCO)].copy()
    a[:, 0] *= aspect
    return a


# --------------------------------------------------------------------------------------------
# Hands
# --------------------------------------------------------------------------------------------
def hand_block(xy: np.ndarray, conf: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
    s = sk.C_LHAND if side == "left" else sk.C_RHAND
    return xy[s], conf[s]


def hand_usable(conf21: np.ndarray, thr: float = HAND_MIN_CONF, min_points: int = HAND_MIN_POINTS) -> bool:
    return int((np.asarray(conf21) >= thr).sum()) >= min_points


def pick_signing_hand(xy: np.ndarray, conf: np.ndarray, previous: Optional[str] = None,
                      thr: float = HAND_MIN_CONF) -> Optional[str]:
    """Which hand is fingerspelling right now: the visible hand that is raised higher
    (fingerspelling happens near the shoulder/face; the other hand rests low).
    A small bonus keeps the previous choice so it does not flip-flop."""
    best, best_score = None, -1e9
    scale = body_scale(xy, conf, thr)
    rest_y = None
    if scale:
        if conf[sk.C_LSH] >= thr and conf[sk.C_RSH] >= thr:
            sh_y = float(xy[sk.C_LSH, 1] + xy[sk.C_RSH, 1]) / 2
        else:
            sh_y = float(xy[sk.C_NOSE, 1]) + 0.6 * scale
        # a hand whose centre is below ~1 shoulder width under the shoulders is resting, not spelling
        rest_y = sh_y + 1.0 * scale
    for side in ("left", "right"):
        h, c = hand_block(xy, conf, side)
        if not hand_usable(c, thr):
            continue
        good = c >= thr
        cy = float(np.mean(h[good, 1]))
        if rest_y is not None and cy > rest_y:
            continue
        height = -cy                                          # higher on screen = larger
        score = height / (scale or 1.0) + 2.0 * float(np.mean(c))
        if side == previous:
            score += 0.3
        if score > best_score:
            best, best_score = side, score
    return best


def canonical_hand(hand_xy: np.ndarray, is_left: bool) -> np.ndarray:
    """(…, 21, 2) -> wrist-centred, palm-scaled, right-hand canonical coordinates."""
    p = np.asarray(hand_xy, np.float32).copy()
    if is_left:
        p[..., 0] = -p[..., 0]
    p = p - p[..., :1, :]
    palm = np.mean(np.linalg.norm(p[..., list(sk.MCPS), :], axis=-1), axis=-1)
    palm = np.maximum(palm, 1e-6)
    return p / palm[..., None, None]


def _cos_bend(a, b, c):
    v1 = b - a
    v2 = c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return np.sum(v1 * v2, axis=-1) / np.maximum(n, 1e-6)


def letter_features(q: np.ndarray) -> np.ndarray:
    """Canonical hand(s) (N, 21, 2) or (21, 2) from `canonical_hand` -> (N, 90) float32.

    40 coordinates (wrist dropped, it is always 0), 10 fingertip-fingertip distances,
    5 fingertip-wrist distances, 8 thumb-tip to knuckle distances, 15 joint bend cosines,
    10 finger direction vectors (knuckle -> tip, unit), 2 index-minus-middle tip (R crossing)."""
    q = np.asarray(q, np.float32)
    single = q.ndim == 2
    if single:
        q = q[None]
    n = q.shape[0]
    out = [q[:, 1:, :].reshape(n, 40)]
    tips = list(sk.FINGERTIPS)
    for i in range(5):
        for j in range(i + 1, 5):
            out.append(np.linalg.norm(q[:, tips[i]] - q[:, tips[j]], axis=-1)[:, None])
    for t in tips:
        out.append(np.linalg.norm(q[:, t], axis=-1)[:, None])
    for m in (5, 9, 13, 17, 6, 10, 14, 18):
        out.append(np.linalg.norm(q[:, 4] - q[:, m], axis=-1)[:, None])
    for chain in sk.FINGER_CHAINS:
        for k in range(1, 4):
            out.append(_cos_bend(q[:, chain[k - 1]], q[:, chain[k]], q[:, chain[k + 1]])[:, None])
    for base, tip in ((2, 4), (5, 8), (9, 12), (13, 16), (17, 20)):
        v = q[:, tip] - q[:, base]
        out.append(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6))
    out.append(q[:, 8] - q[:, 12])
    f = np.concatenate(out, axis=1).astype(np.float32)
    return f[0] if single else f


N_LETTER_FEATURES = 90


# --------------------------------------------------------------------------------------------
# Body frame (words)
# --------------------------------------------------------------------------------------------
def body_scale(xy: np.ndarray, conf: np.ndarray, thr: float = 0.3) -> Optional[float]:
    """Shoulder width; falls back to 6 x the eye distance when a shoulder is missing
    (eyes ~6.3 cm apart, shoulder keypoints ~35-40 cm apart on an adult)."""
    if conf[sk.C_LSH] >= thr and conf[sk.C_RSH] >= thr:
        s = float(np.linalg.norm(xy[sk.C_LSH] - xy[sk.C_RSH]))
        if s > 1e-6:
            return s
    if conf[1] >= thr and conf[2] >= thr:
        s = float(np.linalg.norm(xy[1] - xy[2])) * 6.0
        return s if s > 1e-6 else None
    return None


def body_center(xy: np.ndarray, conf: np.ndarray, thr: float = 0.3) -> Optional[np.ndarray]:
    if conf[sk.C_LSH] >= thr and conf[sk.C_RSH] >= thr:
        return (xy[sk.C_LSH] + xy[sk.C_RSH]) / 2.0
    if conf[sk.C_NOSE] >= thr:
        s = body_scale(xy, conf, thr)
        if s:
            return xy[sk.C_NOSE] + np.array([0.0, 0.6 * s], np.float32)   # shoulders ~0.6 widths below nose
    return None


def mirror_canon(xy: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror a normalised (centred) canonical frame or sequence: x -> -x and swap sides."""
    perm = np.asarray(sk.MIRROR_PERM)
    m = np.asarray(xy, np.float32)[..., perm, :].copy()
    m[..., 0] = -m[..., 0]
    return m, np.asarray(conf)[..., perm].copy()


def dominant_side(seq_xy: np.ndarray, seq_conf: np.ndarray, thr: float = HAND_MIN_CONF) -> str:
    """Dominant hand of a sequence (T, 69, 2): the one that moves more while visible."""
    energy = {}
    for side, s in (("left", sk.C_LHAND), ("right", sk.C_RHAND)):
        c = seq_conf[:, s.start]
        w = seq_xy[:, s.start]
        vis = c >= thr
        e = float(vis.mean())
        both = vis[1:] & vis[:-1]
        if both.any():
            e += float(np.linalg.norm(np.diff(w, axis=0), axis=-1)[both].sum()) / (
                np.median(np.abs(seq_xy[:, sk.C_LSH, 0] - seq_xy[:, sk.C_RSH, 0])) + 1e-6)
        energy[side] = e
    return "left" if energy["left"] > energy["right"] * 1.1 else "right"


WORD_FEATURES = sk.N_CANON * 2 + 42 * 2 + sk.N_CANON   # 138 global + 84 local hands + 69 mask = 291


def normalize_sequence(seq_xy: np.ndarray, seq_conf: np.ndarray, thr: float = 0.3,
                       dominant: Optional[str] = None) -> Tuple[np.ndarray, str]:
    """(T, 69, 2) + (T, 69) raw isotropic -> (T, WORD_FEATURES) float32, dominant side used.

    Per frame: shoulder-centred / shoulder-width scaled global coordinates (the body frame is
    the median over the sequence, so a nodding head does not shake everything), then each
    hand also in its own wrist-centred palm-scaled frame, then the mask. Left-dominant
    sequences are mirrored first."""
    xy = np.asarray(seq_xy, np.float32)
    cf = np.asarray(seq_conf, np.float32)
    t = xy.shape[0]
    centers, scales = [], []
    for i in range(t):
        c = body_center(xy[i], cf[i], thr)
        s = body_scale(xy[i], cf[i], thr)
        if c is not None and s:
            centers.append(c)
            scales.append(s)
    if centers:
        center = np.median(np.stack(centers), axis=0)
        scale = float(np.median(scales))
    else:  # no body at all: fall back to the visible points' bounding box
        good = cf >= thr
        pts = xy[good] if good.any() else xy.reshape(-1, 2)
        center = pts.mean(axis=0)
        scale = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-3)) / 2.0
    mask = (cf >= thr).astype(np.float32)
    g = (xy - center[None, None, :]) / scale
    side = dominant or dominant_side(xy, cf, thr)
    if side == "left":
        g, mask = mirror_canon(g, mask)
    g = g * mask[..., None]
    local = []
    for s in (sk.C_LHAND, sk.C_RHAND):
        h = g[:, s, :]
        m = mask[:, s]
        q = canonical_hand(h, is_left=False)                # already mirrored globally
        q = np.where(np.isfinite(q), q, 0.0) * m[..., None]
        ok = m.sum(axis=1) >= HAND_MIN_POINTS
        q[~ok] = 0.0
        local.append(q.reshape(t, 42))
    feats = np.concatenate([g.reshape(t, -1)] + local + [mask], axis=1).astype(np.float32)
    return np.clip(feats, -10.0, 10.0), side


def resample_sequence(feats: np.ndarray, length: int) -> np.ndarray:
    """Linear time resampling (T, F) -> (length, F)."""
    t = feats.shape[0]
    if t == length:
        return feats.astype(np.float32)
    if t == 1:
        return np.repeat(feats, length, axis=0).astype(np.float32)
    pos = np.linspace(0, t - 1, length)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, t - 1)
    w = (pos - i0)[:, None]
    return ((1 - w) * feats[i0] + w * feats[i1]).astype(np.float32)
