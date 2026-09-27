"""Keypoint layouts: COCO-WholeBody (RTMW, 133 points), MediaPipe, and our canonical set.

Everything here is plain data (no heavy imports) so tests and training share it.

COCO-WholeBody 133 (what RTMW outputs; names verified against rtmlib 0.0.16
``rtmlib/visualization/skeleton/coco133.py``)::

    0-16    body (COCO-17: nose, l/r eye, l/r ear, l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r knee, l/r ankle)
    17-22   feet
    23-90   face, iBUG-68 order (face-0 .. face-67)
    91-111  LEFT hand  (left_hand_root, left_thumb1..4, left_forefinger1..4, middle, ring, pinky)
    112-132 RIGHT hand (same order)

"left"/"right" are the PERSON's anatomical sides (COCO convention), not image sides.

MediaPipe Hands order (verified against mediapipe 0.10.9 ``solutions/hands.py``
``HandLandmark``): WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP, INDEX_MCP,
INDEX_PIP, INDEX_DIP, INDEX_TIP, MIDDLE 9-12, RING 13-16, PINKY 17-20.
COCO-WholeBody hand order is root, thumb1-4, forefinger1-4, middle1-4, ring1-4,
pinky1-4, i.e. the SAME order: the mapping is the identity (``MP_HAND_TO_COCO``).
"""
from __future__ import annotations

# --- COCO-WholeBody 133 ----------------------------------------------------------
N_WHOLEBODY = 133
FACE0 = 23          # face-k is index 23 + k
LHAND0 = 91
RHAND0 = 112

HAND_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
# MediaPipe hand landmark i  ->  COCO-WholeBody hand keypoint (offset from the hand root)
MP_HAND_TO_COCO = list(range(21))

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP = 0, 4, 5, 8
MIDDLE_MCP, MIDDLE_TIP, RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP = 9, 12, 13, 16, 17, 20
FINGERTIPS = (4, 8, 12, 16, 20)
MCPS = (5, 9, 13, 17)
# (joint before, joint, joint after) for the bend angle at each finger joint
FINGER_CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
              (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
              (5, 9), (9, 13), (13, 17)]

# --- Canonical set (69 points) ------------------------------------------------------
# body: nose, l eye, r eye, l ear, r ear, l sh, r sh, l elbow, r elbow, l wrist, r wrist, l hip, r hip
BODY_COCO = list(range(13))
BODY_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow",
              "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip"]
# face subset, iBUG-68 numbers: brows (outer, middle, inner) x2, nose bridge top + tip, lips
FACE_IBUG = [17, 19, 21, 22, 24, 26, 27, 30, 48, 51, 54, 57, 62, 66]
FACE_NAMES = ["brow_a_outer", "brow_a_mid", "brow_a_inner", "brow_b_inner", "brow_b_mid", "brow_b_outer",
              "nose_bridge", "nose_tip", "mouth_corner_a", "lip_top", "mouth_corner_b", "lip_bottom",
              "lip_inner_top", "lip_inner_bottom"]

CANON_WHOLEBODY = (BODY_COCO + [FACE0 + k for k in FACE_IBUG] +
                   list(range(LHAND0, LHAND0 + 21)) + list(range(RHAND0, RHAND0 + 21)))
N_CANON = len(CANON_WHOLEBODY)            # 69
C_BODY = slice(0, 13)
C_FACE = slice(13, 27)
C_LHAND = slice(27, 48)
C_RHAND = slice(48, 69)
C_NOSE, C_LSH, C_RSH, C_LEL, C_REL, C_LWR, C_RWR, C_LHIP, C_RHIP = 0, 5, 6, 7, 8, 9, 10, 11, 12

# Left/right swap permutation inside the canonical set (used to mirror left-handed signers).
_BODY_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
# FACE_IBUG pairs: 17<->26, 19<->24, 21<->22, 27, 30 self, 48<->54, 51 self, 57 self, 62 self, 66 self
_FACE_SWAP = [5, 4, 3, 2, 1, 0, 6, 7, 10, 9, 8, 11, 12, 13]
MIRROR_PERM = (_BODY_SWAP + [13 + i for i in _FACE_SWAP] + list(range(48, 69)) + list(range(27, 48)))

# --- MediaPipe Holistic -> canonical (for the Kaggle ISLR / fingerspelling data) ----------
# Pose landmarks (mediapipe solutions/pose.py PoseLandmark): NOSE 0, LEFT_EYE 2, RIGHT_EYE 5,
# LEFT_EAR 7, RIGHT_EAR 8, LEFT/RIGHT_SHOULDER 11/12, ELBOW 13/14, WRIST 15/16, HIP 23/24.
MP_POSE_TO_CANON_BODY = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
# FaceMesh (468) index for each FACE_IBUG point. Eyebrow and lip indices appear in mediapipe's
# face_mesh_connections.py (FACEMESH_RIGHT_EYEBROW 70-63-105-66-107, FACEMESH_LEFT_EYEBROW
# 300-293-334-296-336, FACEMESH_LIPS corners 61/291, outer 0/17, inner 13/14); nose 168 (bridge)
# and 4 (tip) follow the usual 68->468 correspondence. "right" in MediaPipe = the person's right,
# which is iBUG 17-21 / 48 (image left for a frontal, non-mirrored face).
MP_FACEMESH_FOR_IBUG = {17: 70, 19: 105, 21: 107, 22: 336, 24: 334, 26: 300, 27: 168, 30: 4,
                        48: 61, 51: 0, 54: 291, 57: 17, 62: 13, 66: 14}
MP_FACE_TO_CANON_FACE = [MP_FACEMESH_FOR_IBUG[k] for k in FACE_IBUG]

# --- Drawing (canonical indices) ------------------------------------------------------------
BODY_EDGES = [(C_LSH, C_RSH), (C_LSH, C_LEL), (C_LEL, C_LWR), (C_RSH, C_REL), (C_REL, C_RWR),
              (C_LSH, C_LHIP), (C_RSH, C_RHIP), (C_LHIP, C_RHIP), (0, 1), (0, 2), (1, 3), (2, 4)]


def canon_edges() -> list:
    edges = list(BODY_EDGES)
    for base in (C_LHAND.start, C_RHAND.start):
        edges += [(base + a, base + b) for a, b in HAND_EDGES]
    # lips outline (corner-top-corner-bottom)
    f = C_FACE.start
    edges += [(f + 8, f + 9), (f + 9, f + 10), (f + 10, f + 11), (f + 11, f + 8), (f + 0, f + 1),
              (f + 1, f + 2), (f + 3, f + 4), (f + 4, f + 5), (f + 6, f + 7)]
    return edges
