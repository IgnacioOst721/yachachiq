import os
import numpy as np
from collections import deque


def open_camera():
    """Open the webcam. If it grabs your phone instead of the built-in camera,
    put ASL_CAMERA=1 (or 2) in front of the command to pick a different one."""
    import cv2
    idx = int(os.environ.get("ASL_CAMERA", "0"))
    return cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)

# Shared helpers used by BOTH the collector and the live translator, so they
# always treat the hand identically. You don't need to edit this file.

# Fingertip landmark numbers: thumb, index, middle, ring, pinky
FINGERTIPS = [4, 8, 12, 16, 20]
MOTION_WINDOW = 12   # how many recent frames we measure movement across (~0.5s)


def extract_shape(hand_landmarks):
    """63 numbers describing the hand SHAPE, independent of where the hand is
    in the frame or how big it looks. (This is the 'still pose' part.)"""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                   dtype=np.float32)
    pts = pts - pts[0]                      # move wrist to the center
    max_val = np.max(np.abs(pts))
    if max_val > 0:
        pts = pts / max_val                # scale so biggest value is 1
    return pts.flatten().tolist()


# Keep the old name working for any older code.
extract_landmarks = extract_shape


def fingertip_xy(hand_landmarks):
    """The 5 fingertip positions in image coordinates (used to measure movement)."""
    return np.array([[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y]
                     for i in FINGERTIPS], dtype=np.float32)   # shape (5, 2)


def motion_features(buffer):
    """How far each of the 5 fingertips traveled across recent frames = 5 numbers.
    Held still -> all near 0.  Moving (J, Z) -> the moving finger is large."""
    if len(buffer) < 2:
        return [0.0] * len(FINGERTIPS)
    arr = np.array(buffer)                       # (frames, 5, 2)
    steps = np.diff(arr, axis=0)                 # change between frames
    dist = np.linalg.norm(steps, axis=2)         # distance per finger per step
    return dist.sum(axis=0).tolist()             # total path length per finger


def new_motion_buffer():
    """A short rolling memory of recent fingertip positions."""
    return deque(maxlen=MOTION_WINDOW)


def feature_names():
    """Column headers: 63 shape numbers + 5 motion numbers."""
    shape = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    motion = ["mthumb", "mindex", "mmiddle", "mring", "mpinky"]
    return shape + motion


# Geometric "hint" features derived from the 63 shape numbers. They make the
# hard-to-tell-apart letters (the fist family, U/V/K/R) easier for the model.
# Used by BOTH training and the live app, so they always match.
_TIPS = [4, 8, 12, 16, 20]      # finger tips
_MCPS = [5, 9, 13, 17]          # finger knuckles


def augment(X):
    """X is a (n, 68) array -> returns (n, 68+21) with geometry features added."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    s = X[:, :63].reshape(n, 21, 3)
    ex = []
    for t in _TIPS:                                    # how extended each finger is
        ex.append(np.linalg.norm(s[:, t, :], axis=1))
    for i in range(5):                                 # spacing between fingertips
        for j in range(i + 1, 5):
            ex.append(np.linalg.norm(s[:, _TIPS[i], :] - s[:, _TIPS[j], :], axis=1))
    for m in _MCPS:                                    # thumb position (fist family)
        ex.append(np.linalg.norm(s[:, 4, :] - s[:, m, :], axis=1))
    ex.append(s[:, 8, 0] - s[:, 12, 0])                # index vs middle (crossing -> R)
    ex.append(s[:, 8, 1] - s[:, 12, 1])
    return np.hstack([X, np.stack(ex, axis=1)]).astype(np.float32)
