"""Turn the ASL-HG photos (10 people, letters A-Z) into canonical 2D hands for training.

Two keypoint sources:
  --source mediapipe (default): the MediaPipe Hands landmarks already stored by
      mp_extract.py --all (no model run needed). Same keypoint family as the v1 data; the
      domain-gap study (domain_gap.py) found MediaPipe and RTMPose-hand give the same letter
      accuracy on these photos (59.1 % vs 59.6 %).
  --source rtmpose: RTMPose-m hand keypoints (the runtime finger refiner). Slow on a busy
      Mac (it took ~4 s per photo while other jobs used the CPU/GPU), so it is optional.
Which hand is shown (left/right) comes from MediaPipe's handedness in both cases.

    .venvs/sign-mediapipe/bin/python training/sign/mp_extract.py --all        # ~10 min
    .venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/aslhg_keypoints.py

Output training/sign/data/aslhg_hands.npz: q (N,21,2) canonical right hands, labels (a-z),
people (P1..P10), split (train/test of the HF release), score (mean keypoint confidence).
Data: ASL-HG, Pranto et al., Mendeley Data doi:10.17632/j4y5w2c8w9.1, CC BY 4.0.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from mp_extract import pad_image  # noqa: E402
from yq.sign import features as F  # noqa: E402
from yq.sign import modelstore, rtm  # noqa: E402

NAMES = [str(i) for i in range(10)] + [chr(ord("a") + i) for i in range(26)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp", default=str(HERE / "runs" / "mp_all.npz"))
    ap.add_argument("--backend", default="cpu", help="cpu is steadier when the GPU is busy with other jobs")
    ap.add_argument("--per-person-letter", type=int, default=10,
                    help="max photos per (person, letter); photos of one take are near-duplicates")
    ap.add_argument("--out", default=str(HERE / "data" / "aslhg_hands.npz"))
    ap.add_argument("--source", default="mediapipe", choices=["mediapipe", "rtmpose"])
    args = ap.parse_args()
    if args.source == "mediapipe":
        return from_mediapipe(args)
    import pyarrow.parquet as pq
    from PIL import Image
    mp = np.load(args.mp)
    handed = {str(k): (str(h), bool(f)) for k, h, f in zip(mp["keys"], mp["handedness"], mp["found"])}
    model = rtm.TopDownPose(modelstore.pose_model_path("rtmpose-hand"), (256, 256), args.backend, threads=2)
    qs, labels, people, splits, scores = [], [], [], [], []
    count: dict = {}
    t0 = time.time()
    for split in ("train", "test"):
        t = pq.read_table(HERE / "data" / "asl_hg" / ("%s.parquet" % split))
        lab = t.column("label").to_pylist()
        col = t.column("image").combine_chunks()
        paths = col.field("path").to_pylist()
        for i, k in enumerate(lab):
            if k < 10:
                continue
            key = "%s/%s" % (split, paths[i])
            side, found = handed.get(key, ("", False))
            pl = (paths[i].split("_")[0], k)
            if not found or count.get(pl, 0) >= args.per_person_letter:
                continue
            count[pl] = count.get(pl, 0) + 1
            rgb = np.array(Image.open(io.BytesIO(col.field("bytes")[i].as_py())).convert("RGB"))
            img, box = pad_image(rgb)
            kp, sc = model(np.ascontiguousarray(img[:, :, ::-1]), box)
            # MediaPipe's label assumes a mirrored selfie; these photos are not mirrored
            qs.append(F.canonical_hand(kp, is_left=(side == "Right")))
            labels.append(NAMES[k])
            people.append(paths[i].split("_")[0])
            splits.append(split)
            scores.append(float(sc.mean()))
            if len(qs) % 250 == 0:
                print("  %d hands  %.0fs" % (len(qs), time.time() - t0), flush=True)
    np.savez_compressed(args.out, q=np.stack(qs).astype(np.float32), labels=np.array(labels),
                        people=np.array(people), split=np.array(splits), score=np.array(scores, np.float32))
    print("saved", args.out, len(qs), "hands from", len(set(people)), "people")


def from_mediapipe(args):
    """Canonical hands straight from mp_all.npz (keys look like 'train/P10_A_1000.jpg')."""
    z = np.load(args.mp)
    qs, labels, people, splits = [], [], [], []
    for key, lm, side, found in zip(z["keys"], z["landmarks"], z["handedness"], z["found"]):
        if not found:
            continue
        split, name = str(key).split("/", 1)
        person, letter = name.split("_")[:2]
        # MediaPipe's label assumes a mirrored selfie; these photos are not mirrored
        qs.append(F.canonical_hand(lm, is_left=(str(side) == "Right")))
        labels.append(letter.lower())
        people.append(person)
        splits.append(split)
    np.savez_compressed(args.out, q=np.stack(qs).astype(np.float32), labels=np.array(labels),
                        people=np.array(people), split=np.array(splits), source="mediapipe")
    print("saved", args.out, len(qs), "hands from", len(set(people)), "people (MediaPipe keypoints)")


if __name__ == "__main__":
    main()
