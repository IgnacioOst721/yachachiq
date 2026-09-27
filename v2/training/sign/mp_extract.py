"""Step 1 of the domain-gap study: MediaPipe Hands landmarks for the ASL-HG test images.

Runs in a SEPARATE small venv because mediapipe 0.10.21 (the last release with the legacy
`mp.solutions.hands` API that v1 used) needs numpy<2, and mediapipe 1.0.1's Tasks API crashed
on this Mac ("graph_service.h: Service is unavailable"):

    uv venv --python 3.10 .venvs/sign-mediapipe
    uv pip install --python .venvs/sign-mediapipe/bin/python "mediapipe==0.10.21" pyarrow pillow
    .venvs/sign-mediapipe/bin/python training/sign/mp_extract.py --per-class 20

Writes training/sign/runs/mp_landmarks.npz: images (padded, BGR), labels, people, landmarks
(pixels), handedness. Step 2 is domain_gap.py (sign venv, RTM models + our classifier).
Same settings as v1 collect_data.py: static_image_mode for photos, 1 hand, confidence 0.5.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARQUET = HERE / "data" / "asl_hg" / "test.parquet"
NAMES = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]


def load_images(per_class: int, seed: int = 0):
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(PARQUET)
    labels = np.array(t.column("label").to_pylist())
    col = t.column("image").combine_chunks()
    paths = col.field("path").to_pylist()
    rng = np.random.default_rng(seed)
    out = []
    for k in range(10, 36):                       # letters only
        by_p: dict = {}
        for i in np.where(labels == k)[0]:
            by_p.setdefault(paths[i].split("_")[0], []).append(int(i))
        chosen = []
        while len(chosen) < per_class and any(by_p.values()):
            for p in sorted(by_p):
                if by_p[p] and len(chosen) < per_class:
                    chosen.append(by_p[p].pop(int(rng.integers(len(by_p[p])))))
        for i in chosen:
            im = Image.open(io.BytesIO(col.field("bytes")[i].as_py())).convert("RGB")
            out.append((NAMES[k].lower(), paths[i].split("_")[0], np.array(im)))
    return out


def pad_image(rgb: np.ndarray, target: int = 256):
    """Long side -> `target` px, then a replicated border of target/2 so the palm detector
    sees context. Returns RGB image and the hand box (x1, y1, x2, y2)."""
    from PIL import Image
    h, w = rgb.shape[:2]
    s = target / max(h, w)
    nw, nh = int(w * s), int(h * s)
    img = np.array(Image.fromarray(rgb).resize((nw, nh), Image.BICUBIC))
    p = target // 2
    img = np.pad(img, ((p, p), (p, p), (0, 0)), mode="edge")
    return img, np.array([p, p, p + nw, p + nh], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "runs" / "mp_landmarks.npz"))
    args = ap.parse_args()
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    items = load_images(args.per_class)
    imgs, boxes, labels, people, lms, hand, found = [], [], [], [], [], [], []
    for n, (lab, person, rgb) in enumerate(items):
        img, box = pad_image(rgb)
        H, W = img.shape[:2]
        r = hands.process(img)
        ok = bool(r.multi_hand_landmarks)
        lm = np.full((21, 2), np.nan, np.float32)
        side = ""
        if ok:
            lm = np.array([[p.x * W, p.y * H] for p in r.multi_hand_landmarks[0].landmark], np.float32)
            side = r.multi_handedness[0].classification[0].label
        imgs.append(img[:, :, ::-1])     # BGR for OpenCV/RTM
        boxes.append(box)
        labels.append(lab)
        people.append(person)
        lms.append(lm)
        hand.append(side)
        found.append(ok)
        if n % 100 == 0:
            print("  %d/%d" % (n, len(items)), flush=True)
    shapes = {im.shape for im in imgs}
    obj = np.empty(len(imgs), dtype=object)
    obj[:] = imgs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, images=obj, boxes=np.stack(boxes), labels=np.array(labels), people=np.array(people),
                        landmarks=np.stack(lms), handedness=np.array(hand), found=np.array(found))
    print("saved %s: %d images, MediaPipe found a hand in %d (%d image shapes)" % (
        args.out, len(imgs), int(np.sum(found)), len(shapes)))


if __name__ == "__main__" and "--all" not in sys.argv:
    main()


def extract_all(parquets: list, out: str) -> None:
    """MediaPipe landmarks + handedness for EVERY letter image of the given parquet files
    (no images saved). Used by aslhg_keypoints.py to know which hand each photo shows."""
    import mediapipe as mp
    import pyarrow.parquet as pq
    from PIL import Image
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    keys, lms, hand, found = [], [], [], []
    for pqf in parquets:
        t = pq.read_table(pqf)
        labels = t.column("label").to_pylist()
        col = t.column("image").combine_chunks()
        paths = col.field("path").to_pylist()
        for i, k in enumerate(labels):
            if k < 10:
                continue
            rgb = np.array(Image.open(io.BytesIO(col.field("bytes")[i].as_py())).convert("RGB"))
            img, _box = pad_image(rgb)
            H, W = img.shape[:2]
            r = hands.process(img)
            lm = np.full((21, 2), np.nan, np.float32)
            side = ""
            if r.multi_hand_landmarks:
                lm = np.array([[p.x * W, p.y * H] for p in r.multi_hand_landmarks[0].landmark], np.float32)
                side = r.multi_handedness[0].classification[0].label
            keys.append("%s/%s" % (Path(pqf).stem, paths[i]))
            lms.append(lm)
            hand.append(side)
            found.append(bool(r.multi_hand_landmarks))
            if len(keys) % 1000 == 0:
                print("  %d done" % len(keys), flush=True)
    np.savez_compressed(out, keys=np.array(keys), landmarks=np.stack(lms), handedness=np.array(hand),
                        found=np.array(found))
    print("saved", out, len(keys), "images; hand found in", int(np.sum(found)))


if __name__ == "__main__" and "--all" in sys.argv:
    # .venvs/sign-mediapipe/bin/python training/sign/mp_extract.py --all
    extract_all([str(HERE / "data" / "asl_hg" / n) for n in ("train.parquet", "test.parquet")],
                str(HERE / "runs" / "mp_all.npz"))
