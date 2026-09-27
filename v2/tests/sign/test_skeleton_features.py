from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from yq.sign import features as F
from yq.sign import skeleton as sk
from yq.sign import synth

MP_HAND_ORDER = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP",
                 "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP",
                 "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                 "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
COCO_HAND_ORDER = ["hand_root", "thumb1", "thumb2", "thumb3", "thumb4", "forefinger1", "forefinger2", "forefinger3",
                   "forefinger4", "middle_finger1", "middle_finger2", "middle_finger3", "middle_finger4",
                   "ring_finger1", "ring_finger2", "ring_finger3", "ring_finger4", "pinky_finger1", "pinky_finger2",
                   "pinky_finger3", "pinky_finger4"]
FINGER = {"THUMB": "thumb", "INDEX": "forefinger", "MIDDLE": "middle_finger", "RING": "ring_finger", "PINKY": "pinky_finger"}


def _finger_and_rank(mp_name: str):
    if mp_name == "WRIST":
        return ("root", 0)
    f = mp_name.split("_")[0]
    rank = {"CMC": 1, "MCP": 1 if f != "THUMB" else 2, "IP": 3, "PIP": 2, "DIP": 3, "TIP": 4}[mp_name.split("_")[-1]]
    return (FINGER[f], rank)


def test_hand_mapping_is_identity_and_semantic():
    """MediaPipe i and COCO-WholeBody hand i name the same joint (so the mapping is the identity)."""
    assert sk.MP_HAND_TO_COCO == list(range(21))
    for i, (mp_name, coco) in enumerate(zip(MP_HAND_ORDER, COCO_HAND_ORDER)):
        finger, rank = _finger_and_rank(mp_name)
        if finger == "root":
            assert coco == "hand_root"
        else:
            assert coco == "%s%d" % (finger, rank), (i, mp_name, coco)


def test_hand_mapping_against_installed_libraries():
    """When rtmlib / mediapipe are installed, check their own tables (the source of truth)."""
    spec = importlib.util.find_spec("rtmlib")
    if spec is None:
        pytest.skip("rtmlib not installed")
    p = Path(spec.origin).parent / "visualization" / "skeleton" / "coco133.py"
    ns: dict = {}
    exec(p.read_text(), ns)
    info = ns["coco133"]["keypoint_info"]
    for side, base in (("left", sk.LHAND0), ("right", sk.RHAND0)):
        for i, name in enumerate(COCO_HAND_ORDER):
            assert info[base + i]["name"] == "%s_%s" % (side, name)
    names = [info[i]["name"] for i in sk.BODY_COCO]
    assert names[5:7] == ["left_shoulder", "right_shoulder"] and names[9:11] == ["left_wrist", "right_wrist"]
    try:
        from mediapipe.python.solutions.hands import HandLandmark   # legacy API (<= 0.10.21)
    except Exception:
        return
    assert [h.name for h in HandLandmark] == MP_HAND_ORDER


def test_mirror_permutation_is_involution_and_swaps_hands():
    perm = np.array(sk.MIRROR_PERM)
    assert sorted(perm.tolist()) == list(range(sk.N_CANON))
    assert (perm[perm] == np.arange(sk.N_CANON)).all()
    assert perm[sk.C_LSH] == sk.C_RSH and perm[sk.C_LHAND.start] == sk.C_RHAND.start


def _rand_hand(rng):
    return synth.OPEN_HAND + rng.normal(0, 0.05, (21, 2)).astype(np.float32)


def test_canonical_hand_translation_scale_mirror():
    rng = np.random.default_rng(0)
    h = _rand_hand(rng) * 50 + np.array([300, 200])
    q = F.canonical_hand(h, is_left=False)
    assert np.allclose(q[0], 0) and np.isclose(np.mean(np.linalg.norm(q[[5, 9, 13, 17]], axis=1)), 1, atol=1e-5)
    assert np.allclose(F.canonical_hand(h * 3.0 + 77, False), q, atol=1e-5)          # translation + scale
    mirrored = h.copy()
    mirrored[:, 0] = 1000 - mirrored[:, 0]                                           # the same shape, other hand
    assert np.allclose(F.canonical_hand(mirrored, is_left=True), q, atol=1e-5)
    f1, f2 = F.letter_features(q), F.letter_features(F.canonical_hand(h * 0.5 + 5, False))
    assert f1.shape == (F.N_LETTER_FEATURES,) and np.allclose(f1, f2, atol=1e-4)


def test_normalize_sequence_invariances_and_mirroring():
    rng = np.random.default_rng(1)
    cls = next(c for c in range(20) if not synth.two_handed(c))      # a one-handed sign
    xy, cf = synth.word_sequence(cls, 8, rng)
    f1, side1 = F.normalize_sequence(xy, cf)
    f2, _ = F.normalize_sequence(xy * 1.7 + np.array([55.0, -20.0]), cf)
    assert np.allclose(f1, f2, atol=1e-4)
    # the same sign done with the other hand (a left-handed signer) gives the same features
    m = xy[:, sk.MIRROR_PERM].copy()
    m[..., 0] = 2000 - m[..., 0]
    f3, side3 = F.normalize_sequence(m, cf[:, sk.MIRROR_PERM])
    assert side1 != side3
    assert np.allclose(f1, f3, atol=1e-3)
    assert f1.shape[1] == F.WORD_FEATURES


def test_missing_points_are_masked_zero():
    xy, cf = synth.body()
    cf = cf.copy()
    cf[sk.C_LHAND] = 0.0
    f, _ = F.normalize_sequence(xy[None], cf[None])
    mask = f[0, -sk.N_CANON:]
    assert (mask[sk.C_LHAND] == 0).all() and (mask[sk.C_RHAND] == 1).all()
    g = f[0, : sk.N_CANON * 2].reshape(sk.N_CANON, 2)
    assert (g[sk.C_LHAND] == 0).all()


def test_mediapipe_holistic_mapping_fixes_swapped_hands():
    xy, cf = synth.body()
    pose = np.full((33, 2), np.nan, np.float32)
    for c, m in zip(range(13), sk.MP_POSE_TO_CANON_BODY):
        pose[m] = xy[c]
    face = np.zeros((468, 2), np.float32)
    lh, rh = xy[sk.C_LHAND].copy(), xy[sk.C_RHAND].copy()
    out, ok = F.from_mp_holistic(pose, face, rh, lh)          # deliberately swapped names
    assert np.allclose(out[sk.C_LHAND], lh) and np.allclose(out[sk.C_RHAND], rh)
    assert ok[sk.C_LSH] == 1 and np.allclose(out[sk.C_LSH], xy[sk.C_LSH])


def test_pick_signing_hand_ignores_resting_hand():
    xy, cf = synth.body()
    assert F.pick_signing_hand(xy, cf) is None                   # both hands down
    s = float(np.linalg.norm(xy[sk.C_LSH] - xy[sk.C_RSH]))
    synth.place_hand(xy, cf, "right", synth.OPEN_HAND, xy[sk.C_RSH] + np.array([-20, -10]), 0.28 * s)
    assert F.pick_signing_hand(xy, cf) == "right"
