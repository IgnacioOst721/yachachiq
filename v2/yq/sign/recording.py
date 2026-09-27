"""Saving and checking keypoint recordings (used by record.py, training and tests; no GUI)."""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

from . import features as F
from . import modelstore
from . import skeleton as sk

MOTION_LETTERS_ALL = ("j", "z", "ñ")


def slug(label: str) -> str:
    s = unicodedata.normalize("NFD", label.lower().replace("ñ", "nh"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9_-]+", "_", s).strip("_") or "x"


def take_path(lang: str, mode: str, label: str, signer: str, session: str, root: Optional[Path] = None) -> Path:
    root = Path(root) if root else modelstore.recordings_dir()
    d = root / lang / mode / slug(label)
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("%s_%s_*.npz" % (slug(signer), slug(session)))))
    return d / ("%s_%s_%03d.npz" % (slug(signer), slug(session), n + 1))


def save_take(path: Path, xy, conf, t, label: str, signer: str, session: str, lang: str, mode: str,
              image_wh=(1280, 800), side: str = "right", pose_model: str = "", quality: Optional[dict] = None,
              kpts133=None, scores133=None) -> Path:
    extra = {}
    if kpts133 is not None:
        extra = {"kpts133": np.asarray(kpts133, np.float16), "scores133": np.asarray(scores133, np.float16)}
    np.savez_compressed(path, xy=np.asarray(xy, np.float32), conf=np.asarray(conf, np.float32),
                        t=np.asarray(t, np.float64), label=label, signer=signer, session=session, lang=lang,
                        mode=mode, side=side, image_wh=np.asarray(image_wh), pose_model=pose_model,
                        quality=json.dumps(quality or {}), saved=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
    return path


def check_quality(xy: np.ndarray, conf: np.ndarray, t: np.ndarray, mode: str, label: str = "",
                  thr: float = 0.3) -> tuple:
    """-> (ok, problems_es list, stats dict). Rules are simple on purpose (explainable)."""
    xy, conf, t = np.asarray(xy), np.asarray(conf), np.asarray(t)
    problems = []
    n = len(t)
    dur = float(t[-1] - t[0]) if n > 1 else 0.0
    fps = (n - 1) / dur if dur > 0 else 0.0
    body = conf[:, [sk.C_LSH, sk.C_RSH, sk.C_NOSE]].min(axis=1) >= thr
    hands = np.array([[F.hand_usable(conf[i, s]) for s in (sk.C_LHAND, sk.C_RHAND)] for i in range(n)]) if n else np.zeros((0, 2), bool)
    any_hand = hands.any(axis=1) if n else np.zeros(0, bool)
    stats = {"frames": n, "seconds": round(dur, 2), "fps": round(fps, 1), "body_visible": float(body.mean()) if n else 0.0,
             "hand_visible": float(any_hand.mean()) if n else 0.0,
             "mean_conf_hands": float(conf[:, sk.C_LHAND.start:].mean()) if n else 0.0}
    if n < 8 or fps < 12:
        problems.append("La cámara va muy lenta (%.0f cuadros por segundo). Revisa la luz o el cable." % fps)
    if stats["body_visible"] < 0.8:
        problems.append("No se ve bien tu cuerpo: colócate de frente, con hombros y cara dentro del cuadro.")
    need = 0.8 if mode == "letters" else 0.6
    if stats["hand_visible"] < need:
        problems.append("No se ven bien tus manos. Acércate a la luz y no las saques del cuadro.")
    if mode == "words" and n > 2:
        s = np.median([F.body_scale(xy[i], conf[i]) or np.nan for i in range(n)])
        moves = 0.0
        for side in (sk.C_LHAND, sk.C_RHAND):
            w = xy[:, side.start]
            ok = conf[:, side.start] >= thr
            if ok.sum() > 2:
                moves = max(moves, float(np.ptp(w[ok, 0]) + np.ptp(w[ok, 1])) / (s or 1.0))
        stats["hand_travel_shoulders"] = round(moves, 2)
        if moves < 0.15:
            problems.append("Casi no hubo movimiento. Haz la seña completa, a velocidad normal.")
        if dur < 0.4:
            problems.append("La seña fue demasiado corta.")
    if mode == "letters" and label not in MOTION_LETTERS_ALL and n > 2:
        side = sk.C_RHAND if hands[:, 1].mean() >= hands[:, 0].mean() else sk.C_LHAND
        w = xy[:, side.start]
        palm = np.median(np.linalg.norm(xy[:, side][:, [5, 9, 13, 17]] - xy[:, side][:, :1], axis=2))
        drift = float(np.ptp(w[:, 0]) + np.ptp(w[:, 1])) / max(float(palm), 1e-6)
        stats["hand_drift_palms"] = round(drift, 2)
        if drift > 2.5:
            problems.append("La mano se movió mucho. Para esta letra, mantenla quieta.")
    return (not problems), problems, stats


def load_take(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
