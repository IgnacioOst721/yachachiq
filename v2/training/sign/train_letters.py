"""Train the fingerspelling letter classifier (2D, works on RTMW hand keypoints).

    cd v2
    .venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_letters.py --lang prl
    .venvs/sign/bin/python -m yq.common.heavylock .venvs/sign/bin/python training/sign/train_letters.py --lang ase

Data: the v1 recordings (see letters_data.py) plus, optionally, letter takes recorded with
``python -m yq.sign.record --mode letters`` (RTMW keypoints; one take = one "batch").

Honest evaluation: 5-fold cross-validation where whole takes (100-frame key presses) are held
out together (StratifiedGroupKFold on take ids), never random rows. We report per-frame
accuracy, "window" accuracy (majority of 10 consecutive frames = ~0.35 s hold, what the live
segmenter sees), the confusion matrix, the pairs R/U/V and I/J, and the accuracy when extra
keypoint noise is added (a stand-in for the MediaPipe -> RTMW gap; see domain_gap.py for the
measured one). All data is from ONE signer (Ignacio), so this is NOT signer-independent.

Outputs: MODELS_DIR/sign/letters_<lang>.{npz,json,onnx} and a copy in yq/sign/models/.
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
V2 = HERE.parents[1]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(HERE))

from letters_data import SOURCES, augment_hands, load_v1_csv, medoid_prototypes  # noqa: E402
from yq.common import config  # noqa: E402
from yq.sign import features as F  # noqa: E402
from yq.sign import modelstore  # noqa: E402

# LSP letters whose handshape differs from ASL (lsp/GUIA_LSP.md, MINEDU guide pp. 69-70: only U;
# the N-tilde is N + movement). ASL-HG photos of these letters are NOT used for LSP.
ASLHG_EXCLUDE = {"prl": ("u",), "ase": (), "ils": ()}

PAIRS = [("r", "u"), ("r", "v"), ("u", "v"), ("i", "j"), ("m", "n"), ("s", "a"), ("e", "s"), ("g", "q"),
         ("k", "p"), ("z", "d")]


def load_recorded_letters(lang: str):
    """Letter takes recorded with yq.sign.record (RTMW keypoints) -> (q, y, take ids as strings)."""
    root = modelstore.recordings_dir() / lang / "letters"
    qs, ys, gs = [], [], []
    if not root.is_dir():
        return None
    for f in sorted(root.rglob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            xy, conf = z["xy"], z["conf"]
            label = str(z["label"])
            side = str(z["side"]) if "side" in z.files else "right"
        s = slice(27, 48) if side == "left" else slice(48, 69)
        for t in range(xy.shape[0]):
            if F.hand_usable(conf[t, s]):
                qs.append(F.canonical_hand(xy[t, s], is_left=(side == "left")))
                ys.append(label)
                gs.append(f.stem)
    if not qs:
        return None
    return np.stack(qs).astype(np.float32), np.array(ys), np.array(gs)


def build_mlp(n_in: int, n_out: int, hidden=(256, 128), dropout=0.2):
    import torch.nn as nn
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def train_mlp(q, y_idx, n_cls, seed=0, epochs=60, noise=0.05, verbose=False):
    import torch
    # a few threads only: on the shared Mac, torch with all cores under load was ~40x slower
    torch.set_num_threads(int(os.environ.get("YQ_TORCH_THREADS", "4")))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f0 = F.letter_features(q)
    mu = f0.mean(0)
    sd = f0.std(0) + 1e-4
    model = build_mlp(f0.shape[1], n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(y_idx, minlength=n_cls).astype(np.float32)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1) / n_cls), dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    yt = torch.tensor(y_idx, dtype=torch.long)
    for ep in range(epochs):
        model.train()
        qa = augment_hands(q, rng, noise=noise)
        xa = torch.tensor((F.letter_features(qa) - mu) / sd, dtype=torch.float32)
        perm = torch.randperm(len(yt))
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(model(xa[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print("   epoch %d loss %.4f" % (ep, tot / len(yt)), flush=True)
    model.eval()
    return model, mu.astype(np.float32), sd.astype(np.float32)


def predict(model, mu, sd, q):
    import torch
    with torch.no_grad():
        x = torch.tensor((F.letter_features(q) - mu) / sd, dtype=torch.float32)
        return torch.softmax(model(x), dim=1).numpy()


def window_majority(pred, groups, win=10):
    """Majority vote over sliding windows of `win` consecutive frames inside each take."""
    out = pred.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for k in range(len(idx)):
            lo = max(0, k - win + 1)
            vals = pred[idx[lo:k + 1]]
            out[idx[k]] = np.bincount(vals).argmax()
    return out


def evaluate(q, y, groups, classes, folds=5, epochs=60, seed=0, source=None):
    from sklearn.model_selection import StratifiedGroupKFold
    y_idx = np.array([classes.index(v) for v in y])
    oof = np.zeros((len(y), len(classes)), np.float32)
    oof_noisy = {s: np.zeros((len(y), len(classes)), np.float32) for s in (0.05, 0.10)}
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    rng = np.random.default_rng(123)
    for k, (tr, te) in enumerate(cv.split(q, y_idx, groups)):
        t0 = time.time()
        model, mu, sd = train_mlp(q[tr], y_idx[tr], len(classes), seed=seed + k, epochs=epochs)
        oof[te] = predict(model, mu, sd, q[te])
        for s in oof_noisy:
            qn = augment_hands(q[te], rng, noise=s, rot_deg=0.0, aspect_jitter=0.0, finger_noise=s / 2)
            oof_noisy[s][te] = predict(model, mu, sd, qn)
        acc = float((oof[te].argmax(1) == y_idx[te]).mean())
        print("  fold %d: %d test frames, %d takes, acc %.3f (%.0fs)" % (k, len(te), len(np.unique(groups[te])), acc,
                                                                      time.time() - t0), flush=True)
    pred = oof.argmax(1)
    take_rows = np.array([not str(g).startswith("hg-") for g in groups])
    res = {
        "split": "StratifiedGroupKFold(%d) over recording takes (100-frame key presses); single signer" % folds,
        "frames": int(len(y)), "takes": int(len(np.unique(groups))),
        "frame_accuracy": float((pred == y_idx).mean()),
        # window / take majority only make sense for takes of ONE letter (v1, recordings), not for
        # ASL-HG groups (= a person, all letters mixed)
        "window10_accuracy": float((window_majority(pred, groups) == y_idx)[take_rows].mean()) if take_rows.any() else None,
        "top3_accuracy": float(np.mean([y_idx[i] in np.argsort(-oof[i])[:3] for i in range(len(y))])),
        "noise_frame_accuracy": {str(s): float((v.argmax(1) == y_idx).mean()) for s, v in oof_noisy.items()},
    }
    # I and J (and N / N-tilde) are the same hand shape; the motion tracker separates them live.
    fam = {c: c for c in classes}
    if "i" in fam and "j" in fam:
        fam["j"] = "i"
    fp = np.array([fam[classes[p]] for p in pred])
    ft = np.array([fam[v] for v in y])
    res["frame_accuracy_ij_merged"] = float((fp == ft).mean())
    cm = np.zeros((len(classes), len(classes)), np.int32)
    for a, b in zip(y_idx, pred):
        cm[a, b] += 1
    res["confusion"] = cm.tolist()
    res["per_class_recall"] = {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(classes)}
    res["pairs"] = {}
    for a, b in PAIRS:
        if a in classes and b in classes:
            ia, ib = classes.index(a), classes.index(b)
            res["pairs"]["%s->%s" % (a, b)] = float(cm[ia, ib] / max(cm[ia].sum(), 1))
            res["pairs"]["%s->%s" % (b, a)] = float(cm[ib, ia] / max(cm[ib].sum(), 1))
    # take-level: majority over the whole take
    take_ok = []
    for g in np.unique(groups[take_rows]):
        idx = groups == g
        take_ok.append(np.bincount(pred[idx], minlength=len(classes)).argmax() == np.bincount(y_idx[idx]).argmax())
    res["take_accuracy"] = float(np.mean(take_ok)) if take_ok else None
    if source is not None:
        res["frame_accuracy_by_source"] = {str(src): float((pred[source == src] == y_idx[source == src]).mean())
                                           for src in np.unique(source)}
        res["top3_accuracy_by_source"] = {
            str(src): float(np.mean([y_idx[i] in np.argsort(-oof[i])[:3] for i in np.where(source == src)[0]]))
            for src in np.unique(source)}
    return res, oof


def baseline_extratrees(raw, y, groups, folds=5, seed=0):
    """v1's recipe (ExtraTrees on the 63 MediaPipe 3D numbers), same folds, for comparison."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(raw, y, groups):
        clf = ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=4).fit(raw[tr], y[tr])
        pred[te] = clf.predict(raw[te])
    return float((pred == y).mean())


def export(model, mu, sd, classes, meta, out_dir: Path, lang: str):
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {"mu": mu, "sd": sd}
    lin = [m for m in model if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        weights["W%d" % i] = m.weight.detach().numpy().astype(np.float32)
        weights["b%d" % i] = m.bias.detach().numpy().astype(np.float32)
    npz = out_dir / ("letters_%s.npz" % lang)
    np.savez(npz, **weights)
    (out_dir / ("letters_%s.json" % lang)).write_text(json.dumps(meta, indent=1, ensure_ascii=False))

    class WithNorm(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.register_buffer("mu", torch.tensor(mu))
            self.register_buffer("sd", torch.tensor(sd))

        def forward(self, x):
            return torch.softmax(self.net((x - self.mu) / self.sd), dim=1)

    wrapped = WithNorm(model).eval()
    onnx_path = out_dir / ("letters_%s.onnx" % lang)
    dummy = torch.zeros(1, len(mu))
    torch.onnx.export(wrapped, (dummy,), str(onnx_path), input_names=["features"], output_names=["probs"],
                      dynamic_axes={"features": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
    return npz, onnx_path, wrapped


def parity(npz: Path, onnx_path: Path, wrapped, n_feat: int) -> dict:
    import onnxruntime as ort
    import torch
    from yq.sign.letters import LetterClassifier
    x = np.random.default_rng(0).normal(0, 1, (64, n_feat)).astype(np.float32) * 2
    with torch.no_grad():
        pt = wrapped(torch.tensor(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ox = sess.run(None, {"features": x})[0]
    npy = LetterClassifier.load("", path=npz).predict_proba(x)
    return {"max_abs_onnx_vs_torch": float(np.abs(ox - pt).max()), "max_abs_numpy_vs_torch": float(np.abs(npy - pt).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="prl", choices=["prl", "ase", "ils"])
    ap.add_argument("--csv", default=None, help="v1 CSV (default: the known path for the language)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--aslhg", default=None, help="add ASL-HG hands (npz from aslhg_keypoints.py): 10 more people")
    args = ap.parse_args()
    lang = args.lang

    parts_q, parts_y, parts_g, raw = [], [], [], None
    csv_path = Path(args.csv) if args.csv else SOURCES.get(lang)
    sources = []
    if csv_path and Path(csv_path).exists():
        d = load_v1_csv(Path(csv_path))
        parts_q.append(d["q"])
        parts_y.append(d["y"])
        parts_g.append(np.array(["v1-%d" % b for b in d["batch"]]))
        raw = (d["raw"], d["y"], parts_g[-1])
        sources.append(str(csv_path))
        print("v1 data: %d frames, %d takes, %d labels from %s" % (len(d["y"]), len(np.unique(d["batch"])),
                                                                  len(set(d["y"])), csv_path))
    if args.aslhg:
        z = np.load(args.aslhg)
        keep = np.array([lab not in ASLHG_EXCLUDE.get(lang, ()) for lab in z["labels"]])
        parts_q.append(z["q"][keep])
        parts_y.append(z["labels"][keep])
        parts_g.append(np.array(["hg-" + p for p in z["people"][keep]]))
        sources.append("ASL-HG (CC BY 4.0), %d hands, %d people" % (int(keep.sum()), len(set(z["people"]))))
        print("ASL-HG: %d hands (excluded for %s: %s)" % (int(keep.sum()), lang, ASLHG_EXCLUDE.get(lang, ())))
    rec = load_recorded_letters(lang)
    if rec is not None:
        parts_q.append(rec[0])
        parts_y.append(rec[1])
        parts_g.append(np.array(["rec-" + g for g in rec[2]]))
        sources.append(str(modelstore.recordings_dir() / lang / "letters"))
        print("recorded RTMW takes: %d frames" % len(rec[1]))
    if not parts_q:
        print("no data for %s" % lang)
        return 1
    q = np.concatenate(parts_q)
    y = np.concatenate(parts_y)
    groups = np.concatenate(parts_g)
    classes = sorted(set(y), key=lambda c: (len(c) > 1, c))

    print("\n== cross-validation (%d folds, whole takes held out) ==" % args.folds)
    source = np.array([g.split("-")[0] for g in groups])      # v1 / hg / rec
    res, oof = evaluate(q, y, groups, classes, folds=args.folds, epochs=args.epochs, source=source)
    if "hg" in set(source):
        res["note_hg"] = ("ASL-HG rows are grouped by PERSON, so their out-of-fold accuracy is on people the "
                          "model never saw (signer-independent); v1 rows are grouped by take (same signer)")
    if raw is not None and not args.no_baseline:
        res["baseline_v1_extratrees_3d_frame_accuracy"] = baseline_extratrees(raw[0], raw[1], raw[2], args.folds)
    print(json.dumps({k: v for k, v in res.items() if k not in ("confusion", "per_class_recall")}, indent=1))
    weak = sorted(res["per_class_recall"].items(), key=lambda kv: kv[1])[:8]
    print("weakest letters:", ", ".join("%s %.2f" % kv for kv in weak))

    print("\n== final model on all data ==")
    y_idx = np.array([classes.index(v) for v in y])
    model, mu, sd = train_mlp(q, y_idx, len(classes), seed=42, epochs=args.epochs, verbose=True)
    protos = medoid_prototypes(q, y)
    meta = {
        "lang": lang, "classes": classes, "features": "letter_features_v1", "n_features": F.N_LETTER_FEATURES,
        "hidden": [256, 128], "trained": time.strftime("%Y-%m-%d %H:%M"), "sources": sources,
        "v1_aspect": 16 / 9, "metrics": res, "prototypes": protos,
    }
    out_dir = Path(args.out) if args.out else modelstore.sign_dir()
    npz, onnx_path, wrapped = export(model, mu, sd, classes, meta, out_dir, lang)
    try:
        par = parity(npz, onnx_path, wrapped, F.N_LETTER_FEATURES)
        meta["parity"] = par
        (out_dir / ("letters_%s.json" % lang)).write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        print("parity:", par)
    except Exception as e:  # onnxruntime missing etc.
        print("parity check skipped:", e)
    modelstore.PACKAGE_MODELS.mkdir(parents=True, exist_ok=True)
    for ext in (".npz", ".json", ".onnx"):
        src = out_dir / ("letters_%s%s" % (lang, ext))
        if src.exists() and src.parent.resolve() != modelstore.PACKAGE_MODELS.resolve():
            shutil.copy2(src, modelstore.PACKAGE_MODELS / src.name)
    print("saved", npz, onnx_path)
    return 0


if __name__ == "__main__":
    config.ensure_dirs()
    sys.exit(main())
