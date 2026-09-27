"""Step 2 of the domain-gap study: MediaPipe vs RTM hand keypoints on real photos of 10 people.

Data: ASL-HG (Pranto et al., Mendeley Data doi:10.17632/j4y5w2c8w9.1, CC BY 4.0; HF mirror
juanjodurillo/asl-hg), test split: tight crops of hands, 10 participants P1-P10, 0-9 + A-Z.
Step 1 (mp_extract.py, separate venv) stored the images + MediaPipe landmarks.

For every image we run RTMPose-m hand (the finger refiner in pose.py) and RTMW-m whole body
(the runtime model; on hand-only crops it is out of distribution, so its numbers here are a
pessimistic bound) and compare against MediaPipe in canonical hand units (wrist-centred,
divided by palm size). Then the letter model trained on Ignacio's v1 data (letters_ase)
classifies each version: accuracy with MediaPipe input = generalisation to 10 NEW signers;
the drop to RTM input = the effect of the keypoint domain gap.

    .venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/domain_gap.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
sys.path.insert(0, str(V2))

from yq.sign import features as F  # noqa: E402
from yq.sign import modelstore, rtm  # noqa: E402
from yq.sign.letters import LetterClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp", default=str(HERE / "runs" / "mp_landmarks.npz"))
    ap.add_argument("--model", default="ase", help="letter model language")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--out", default=str(HERE / "runs" / "domain_gap.json"))
    args = ap.parse_args()
    z = np.load(args.mp, allow_pickle=True)
    imgs, boxes, labels, people = z["images"], z["boxes"], z["labels"], z["people"]
    lms, hand, found = z["landmarks"], z["handedness"], z["found"]
    hand_m = rtm.TopDownPose(modelstore.pose_model_path("rtmpose-hand"), (256, 256), "cpu")
    body_m = rtm.TopDownPose(modelstore.pose_model_path("rtmw-m"), (192, 256), "cpu")
    clf = LetterClassifier.load(args.model, path=args.model_path)
    letters = [c for c in clf.classes if len(c) == 1]
    li = [clf.classes.index(c) for c in letters]
    rows = []
    for n in range(len(imgs)):
        r = {"label": str(labels[n]), "person": str(people[n]), "mp": bool(found[n])}
        if not found[n]:
            rows.append(r)
            continue
        img = imgs[n]
        H, W = img.shape[:2]
        # legacy MediaPipe Hands assumes a mirrored selfie image; these photos are not mirrored,
        # so the label "Left" means the signer's RIGHT hand.
        is_left = str(hand[n]) == "Right"
        hk, hs = hand_m(img, boxes[n])
        bk, bs = body_m(img, np.array([0, 0, W, H], np.float32))
        lb, rb = float(bs[91:112].mean()), float(bs[112:133].mean())
        wk = bk[91:112] if lb >= rb else bk[112:133]
        r.update({"is_left": bool(is_left), "rtmhand_score": float(hs.mean()), "rtmw_hand_score": max(lb, rb)})
        qs = {}
        for name, pts in (("mediapipe", lms[n]), ("rtmpose_hand", hk), ("rtmw_m", wk)):
            q = F.canonical_hand(pts, is_left)
            qs[name] = q
            p = clf.predict_proba(F.letter_features(q)[None])[0][li]
            order = np.argsort(-p)
            r[name] = {"top1": letters[order[0]], "top3": [letters[i] for i in order[:3]]}
        for name in ("rtmpose_hand", "rtmw_m"):
            d = np.linalg.norm(qs[name] - qs["mediapipe"], axis=1)
            r["dist_" + name] = float(d[1:].mean())
            r["dist_tips_" + name] = float(d[[4, 8, 12, 16, 20]].mean())
        rows.append(r)
    ok = [r for r in rows if r["mp"]]
    static = [r for r in ok if r["label"] not in ("j", "z")]
    rep = {"dataset": "ASL-HG test split, letters A-Z: %d images, %d people" % (len(rows), len({r["person"] for r in rows})),
           "mediapipe_found_hand": round(len(ok) / max(len(rows), 1), 3), "static_images_scored": len(static),
           "letter_model": args.model_path or ("letters_%s" % args.model)}
    for name in ("mediapipe", "rtmpose_hand", "rtmw_m"):
        rep["top1_" + name] = round(float(np.mean([r[name]["top1"] == r["label"] for r in static])), 4)
        rep["top3_" + name] = round(float(np.mean([r["label"] in r[name]["top3"] for r in static])), 4)
    for name in ("rtmpose_hand", "rtmw_m"):
        rep["mean_dist_palm_units_" + name] = round(float(np.mean([r["dist_" + name] for r in ok])), 4)
        rep["median_dist_palm_units_" + name] = round(float(np.median([r["dist_" + name] for r in ok])), 4)
        rep["mean_tip_dist_palm_units_" + name] = round(float(np.mean([r["dist_tips_" + name] for r in ok])), 4)
    per = {}
    for r in static:
        per.setdefault(r["label"], []).append(r["mediapipe"]["top1"] == r["label"])
    rep["per_letter_top1_mediapipe"] = {k: round(float(np.mean(v)), 2) for k, v in sorted(per.items())}
    per_p = {}
    for r in static:
        per_p.setdefault(r["person"], []).append(r["mediapipe"]["top1"] == r["label"])
    rep["per_person_top1_mediapipe"] = {k: round(float(np.mean(v)), 2) for k, v in sorted(per_p.items())}
    Path(args.out).write_text(json.dumps({"summary": rep, "rows": rows}, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
