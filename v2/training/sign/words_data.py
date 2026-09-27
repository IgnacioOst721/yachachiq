"""Datasets for the isolated-word model -> list of (feats (T,F), label, signer, session).

Sources
  recordings  MODELS_DIR/sign/recordings/<lang>/words/<label>/*.npz   (python -m yq.sign.record)
  kaggle      Google ISLR "asl-signs" (Kaggle competition data, CC BY 4.0, needs a Kaggle login):
              <dir>/train.csv (path, participant_id, sequence_id, sign) and
              <dir>/train_landmark_files/<participant>/<sequence>.parquet with columns
              frame, row_id, type (face/left_hand/pose/right_hand), landmark_index, x, y, z
  synthetic   synth.word_sequence (pipeline test only, says nothing about accuracy)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from yq.sign import features as F  # noqa: E402
from yq.sign import modelstore, synth  # noqa: E402


def from_recordings(lang: str, root: Path = None) -> list:
    root = Path(root) if root else modelstore.recordings_dir() / lang / "words"
    out = []
    for f in sorted(root.rglob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            feats, _side = F.normalize_sequence(z["xy"], z["conf"])
            out.append((feats, str(z["label"]), str(z["signer"]), str(z["session"])))
    return out


def parquet_to_canon(path: Path, aspect: float = 1.0):
    """One ISLR parquet -> (xy (T,69,2), conf (T,69)); NaN landmarks -> conf 0."""
    import pandas as pd
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y"])
    frames = np.sort(df["frame"].unique())
    parts = {}
    for typ, n in (("pose", 33), ("face", 468), ("left_hand", 21), ("right_hand", 21)):
        sub = df[df["type"] == typ]
        arr = np.full((len(frames), n, 2), np.nan, np.float32)
        fi = np.searchsorted(frames, sub["frame"].to_numpy())
        li = sub["landmark_index"].to_numpy().astype(int)
        arr[fi, li, 0] = sub["x"].to_numpy()
        arr[fi, li, 1] = sub["y"].to_numpy()
        parts[typ] = arr
    xs, cs = [], []
    for t in range(len(frames)):
        lh = parts["left_hand"][t]
        rh = parts["right_hand"][t]
        xy, ok = F.from_mp_holistic(parts["pose"][t], parts["face"][t],
                                    None if np.isnan(lh).all() else lh, None if np.isnan(rh).all() else rh, aspect)
        xs.append(xy)
        cs.append(ok)
    return np.stack(xs), np.stack(cs)


def from_kaggle(kdir: Path, limit: int = 0, aspect: float = 1.0, cache: Path = None) -> list:
    kdir = Path(kdir)
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        return list(zip(z["feats"], z["labels"], z["signers"], z["sessions"]))
    rows = list(csv.DictReader(open(kdir / "train.csv")))
    if limit:
        rows = rows[:limit]
    out = []
    for i, r in enumerate(rows):
        xy, cf = parquet_to_canon(kdir / r["path"], aspect)
        feats, _ = F.normalize_sequence(xy, cf)
        out.append((feats, r["sign"], str(r["participant_id"]), str(r["sequence_id"])))
        if i % 2000 == 0:
            print("  kaggle %d/%d" % (i, len(rows)), flush=True)
    if cache:
        obj = np.empty(len(out), dtype=object)
        obj[:] = [o[0] for o in out]
        np.savez(cache, feats=obj, labels=np.array([o[1] for o in out]), signers=np.array([o[2] for o in out]),
                 sessions=np.array([o[3] for o in out]))
    return out


def synthetic(n_classes: int = 8, per_class: int = 30, signers: int = 5, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n_classes):
        for k in range(per_class):
            xy, cf = synth.word_sequence(c, n_classes, rng)
            feats, _ = F.normalize_sequence(xy, cf)
            out.append((feats, "SIGN%02d" % c, "s%d" % (k % signers), "sess%d" % k))
    return out


def augment_clip(feats: np.ndarray, frames: int, rng: np.random.Generator, train: bool) -> np.ndarray:
    """Random temporal crop (keep 80-100 %) + resample + coordinate noise + frame dropout."""
    t = feats.shape[0]
    if train and t > 4:
        keep = int(round(t * rng.uniform(0.8, 1.0)))
        start = int(rng.integers(0, t - keep + 1))
        feats = feats[start:start + keep]
    x = F.resample_sequence(feats, frames)
    if train:
        n_coord = x.shape[1] - 69                  # last 69 columns are the mask
        noise = rng.normal(0, 0.02, (frames, n_coord)).astype(np.float32)
        x[:, :n_coord] += noise * (x[:, :n_coord] != 0)
        s = rng.uniform(0.9, 1.1)
        x[:, :n_coord] *= s
        drop = rng.random(frames) < 0.05
        x[drop] = 0.0
    return x
