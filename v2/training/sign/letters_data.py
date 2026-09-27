"""Load the v1 fingerspelling recordings (MediaPipe Hands CSV) as canonical 2D hands.

v1 CSV format (lsp/collect_data.py + lsp/landmark_utils.py): ``label, x0,y0,z0 ... x20,y20,z20,
mthumb, mindex, mmiddle, mring, mpinky``. The 63 numbers are MediaPipe Hands landmarks of a
MIRRORED camera frame (collect_data.py does ``cv2.flip(frame, 1)``), wrist-subtracted and
divided by the largest absolute value; x is in units of image WIDTH and y of image HEIGHT.

To get isotropic geometry we multiply x by the camera aspect W/H. The recordings were made on
the MacBook webcam (16:9). A bone-length consistency check over the whole file (rigid palm
bones should keep the same proportions whatever the hand orientation) is minimised near
W/H = 2.0-2.1 and is almost as good at 16/9 = 1.78 but clearly worse at 4/3 (see README), so we
use 16/9 and train with +-15 % aspect jitter.

Handedness: in the mirrored frames Ignacio's right hand looks like a left hand (letter L:
thumb tip to the image LEFT of the wrist), so every v1 sample is mirrored back
(``is_left=True``) into our canonical right hand.

Each key press recorded 100 consecutive frames = one "batch" (one recording take). Batches are
the unit for honest held-out evaluation (frames inside a take are near-duplicates).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[2]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from yq.sign.features import canonical_hand  # noqa: E402

V1_ASPECT = 16.0 / 9.0
REPO = V2.parent
SOURCES = {
    "prl": REPO / "lsp" / "lsp_data.csv",
    "ase": Path.home() / "asl-camera" / "asl_data.csv",
}
BATCH = 100


def load_v1_csv(path: Path, aspect: float = V1_ASPECT):
    """-> dict(q=(N,21,2) canonical right hands, y=(N,) labels, batch=(N,) take ids, raw=(N,63))."""
    rows = [r for r in csv.reader(open(path)) if r][1:]
    labels = np.array([r[0] for r in rows])
    X = np.array([[float(v) for v in r[1:64]] for r in rows], np.float32)
    P = X.reshape(-1, 21, 3)[:, :, :2].copy()
    P[:, :, 0] *= aspect
    q = canonical_hand(P, is_left=True)
    # batches: consecutive runs of the same label, cut every 100 rows
    batch = np.zeros(len(labels), np.int32)
    b = -1
    run_pos = 0
    for i, lab in enumerate(labels):
        if i == 0 or lab != labels[i - 1]:
            run_pos = 0
        if run_pos % BATCH == 0:
            b += 1
        batch[i] = b
        run_pos += 1
    return {"q": q.astype(np.float32), "y": labels, "batch": batch, "raw": X}


def augment_hands(q: np.ndarray, rng: np.random.Generator, noise: float = 0.05, rot_deg: float = 12.0,
                  aspect_jitter: float = 0.15, finger_noise: float = 0.04) -> np.ndarray:
    """Random rotation, aspect (x/y) jitter, per-point Gaussian noise (palm units) and small
    whole-finger offsets, then re-normalised. Simulates a different camera and a different
    keypoint model (RTMW) than the one used for recording."""
    n = q.shape[0]
    a = np.deg2rad(rng.uniform(-rot_deg, rot_deg, n))
    c, s = np.cos(a), np.sin(a)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)       # (n,2,2)
    p = np.einsum("nij,nkj->nki", R, q)
    p[:, :, 0] *= rng.uniform(1 - aspect_jitter, 1 + aspect_jitter, n)[:, None]
    p = p + rng.normal(0, noise, p.shape)
    # correlated offset of whole fingers (keypoint models tend to shift a finger as a unit)
    for f0 in (1, 5, 9, 13, 17):
        p[:, f0:f0 + 4, :] += rng.normal(0, finger_noise, (n, 1, 2))
    return canonical_hand(p.astype(np.float32), is_left=False)


def medoid_prototypes(q: np.ndarray, y: np.ndarray, max_n: int = 400, seed: int = 0) -> dict:
    """One REAL example hand per letter: the medoid (smallest summed distance to the others).
    A coordinate-wise median can be a hand that never existed (orientations average out)."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in sorted(set(y)):
        idx = np.where(y == c)[0]
        if len(idx) > max_n:
            idx = rng.choice(idx, max_n, replace=False)
        f = q[idx].reshape(len(idx), -1)
        d = np.linalg.norm(f[:, None, :] - f[None, :, :], axis=-1).sum(1)
        out[c] = q[idx[int(np.argmin(d))]].round(4).tolist()
    return out
