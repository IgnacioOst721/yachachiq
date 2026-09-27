"""Train the isolated-word model (skeleton transformer) and export ONNX for the Jetson.

    # our own recordings (python -m yq.sign.record --mode words ...)
    .venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_words.py --lang prl
    # Google ISLR asl-signs (after downloading it from Kaggle, see README)
    .venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_words.py \
        --lang ase --source kaggle --kaggle-dir training/sign/data/asl-signs
    # pipeline check on synthetic data (no accuracy meaning)
    .venvs/sign/bin/python training/sign/train_words.py --lang ils --source synthetic --out /tmp/x

Validation is signer-independent when there are >= 3 signers (GroupKFold by signer, or one
fold with --holdout-signers), else by recording session. Reports top-1 / top-5.
Outputs words_<lang>.onnx + .json (classes, frames, metrics, gloss->text) in MODELS_DIR/sign.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import words_data as WD  # noqa: E402
from yq.common import config  # noqa: E402
from yq.sign import features as F  # noqa: E402
from yq.sign import modelstore  # noqa: E402

FRAMES = 32


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def fit(items, classes, epochs=60, seed=0, verbose=False, d=192, layers=3):
    import torch
    # a few threads only: on the shared Mac, torch with all cores under load was ~40x slower
    torch.set_num_threads(int(os.environ.get("YQ_TORCH_THREADS", "4")))
    from word_model import SignTransformer
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = SignTransformer(F.WORD_FEATURES, len(classes), FRAMES, d=d, layers=layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps = epochs * max(1, (len(items) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    yi = np.array([classes.index(it[1]) for it in items])
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(items))
        tot = 0.0
        for i in range(0, len(order), 64):
            b = order[i:i + 64]
            x = torch.tensor(np.stack([WD.augment_clip(items[j][0], FRAMES, rng, True) for j in b])).to(dev)
            y = torch.tensor(yi[b]).to(dev)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print("   epoch %d loss %.4f" % (ep, tot / len(items)), flush=True)
    return model.eval().cpu()


def predict(model, items):
    import torch
    rng = np.random.default_rng(0)
    x = torch.tensor(np.stack([WD.augment_clip(it[0], FRAMES, rng, False) for it in items]))
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).numpy()


def split_groups(items, holdout: int):
    signers = sorted({it[2] for it in items})
    if len(signers) >= 3:
        return "signer", np.array([it[2] for it in items])
    return "session", np.array([it[3] for it in items])


def evaluate(items, classes, folds, epochs):
    from sklearn.model_selection import GroupKFold
    kind, groups = split_groups(items, 0)
    y = np.array([classes.index(it[1]) for it in items])
    n_groups = len(set(groups))
    k = min(folds, n_groups)
    if k < 2:
        return {"split": "none (only one %s)" % kind}
    probs = np.zeros((len(items), len(classes)), np.float32)
    for f, (tr, te) in enumerate(GroupKFold(n_splits=k).split(items, y, groups)):
        t0 = time.time()
        m = fit([items[i] for i in tr], classes, epochs=epochs, seed=f)
        probs[te] = predict(m, [items[i] for i in te])
        print("  fold %d: %d test, top1 %.3f (%.0fs)" % (f, len(te), (probs[te].argmax(1) == y[te]).mean(),
                                                         time.time() - t0), flush=True)
    top5 = np.argsort(-probs, 1)[:, :5]
    return {"split": "GroupKFold(%d) by %s (%d %ss): no %s in both train and test" % (k, kind, n_groups, kind, kind),
            "samples": len(items), "classes": len(classes),
            "top1": float((top5[:, 0] == y).mean()), "top5": float(np.mean([y[i] in top5[i] for i in range(len(y))]))}


def export(model, classes, meta, out_dir: Path, lang: str):
    import onnxruntime as ort
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / ("words_%s.onnx" % lang)
    dummy = torch.zeros(1, FRAMES, F.WORD_FEATURES)
    torch.onnx.export(model, (dummy,), str(onnx_path), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "b"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
    x = np.random.default_rng(0).normal(0, 0.5, (8, FRAMES, F.WORD_FEATURES)).astype(np.float32)
    x[:, -3:] = 0.0                                     # some empty frames (mask path)
    with torch.no_grad():
        pt = model(torch.tensor(x)).numpy()
    ox = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    meta["parity_max_abs_logit"] = float(np.abs(pt - ox).max())
    (out_dir / ("words_%s.json" % lang)).write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return onnx_path, meta["parity_max_abs_logit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="prl")
    ap.add_argument("--source", default="recordings", choices=["recordings", "kaggle", "synthetic"])
    ap.add_argument("--kaggle-dir", default=str(HERE / "data" / "asl-signs"))
    ap.add_argument("--kaggle-aspect", type=float, default=1.0, help="image W/H of the Kaggle videos")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--small", action="store_true", help="d=96, 2 layers (tests)")
    args = ap.parse_args()
    if args.source == "recordings":
        items = WD.from_recordings(args.lang)
    elif args.source == "kaggle":
        items = WD.from_kaggle(Path(args.kaggle_dir), args.limit, args.kaggle_aspect,
                               cache=HERE / "data" / "asl_signs_feats.npz")
    else:
        items = WD.synthetic()
    if not items:
        print("no data: record signs first (python -m yq.sign.record --mode words)")
        return 1
    classes = sorted({it[1] for it in items})
    print("%d samples, %d classes, %d signers" % (len(items), len(classes), len({it[2] for it in items})))
    metrics = {} if args.no_cv else evaluate(items, classes, args.folds, args.epochs)
    print(json.dumps(metrics, indent=1))
    kw = {"d": 96, "layers": 2} if args.small else {}
    model = fit(items, classes, epochs=args.epochs, seed=42, verbose=True, **kw)
    text = {c: c for c in classes}
    meta = {"lang": args.lang, "classes": classes, "frames": FRAMES, "n_features": F.WORD_FEATURES,
            "source": args.source, "trained": time.strftime("%Y-%m-%d %H:%M"), "metrics": metrics, "text": text}
    out_dir = Path(args.out) if args.out else modelstore.sign_dir()
    onnx_path, par = export(model, classes, meta, out_dir, args.lang)
    print("saved", onnx_path, "parity", par)
    if args.out is None and onnx_path.stat().st_size < 20e6:
        for ext in (".onnx", ".json"):
            shutil.copy2(out_dir / ("words_%s%s" % (args.lang, ext)), modelstore.PACKAGE_MODELS)
    return 0


if __name__ == "__main__":
    config.ensure_dirs()
    sys.exit(main())
