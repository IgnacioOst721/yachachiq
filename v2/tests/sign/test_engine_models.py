from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from yq.common.contracts import SignToken
from yq.sign import SignEngine, available, modelstore, sources, synth
from yq.sign import recording as R
from yq.sign.letters import LetterClassifier

V2 = Path(__file__).resolve().parents[2]


def _spell(eng, words):
    shapes = {k: np.asarray(v, np.float32) for k, v in eng.speller.clf.prototypes.items()}
    src = sources.SyntheticSource(words=words, shapes=shapes, realtime=False, loop=False)
    while True:
        g = src.next()
        if g is None:
            return
        eng.process_keypoints(*g)


def test_available_lists_three_sign_languages():
    langs = {d["code"]: d for d in available()}
    assert set(langs) == {"prl", "ase", "ils"}
    assert langs["prl"]["letters"] and langs["ase"]["letters"]
    assert all(set(d) == {"code", "name_es", "letters", "words"} for d in langs.values())


@pytest.mark.parametrize("lang", ["prl", "ase"])
def test_letter_models_load_and_are_sane(lang):
    clf = LetterClassifier.load(lang)
    assert {"a", "b", "l", "y"} <= set(clf.classes)
    q = np.stack([np.asarray(v, np.float32) for v in clf.prototypes.values()])
    from yq.sign import features as F
    p = clf.predict_proba(F.letter_features(q))
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    # each stored prototype (a real training hand) is recognised as its own letter
    names = list(clf.prototypes)
    assert np.mean([clf.classes[i] == n for i, n in zip(p.argmax(1), names)]) > 0.9


@pytest.mark.parametrize("lang", ["prl", "ase"])
def test_letter_model_onnx_parity(lang):
    ort = pytest.importorskip("onnxruntime")
    clf = LetterClassifier.load(lang)
    onnx = modelstore.find_model("letters_%s.onnx" % lang)
    x = np.random.default_rng(0).normal(0, 1, (32, len(clf.mu))).astype(np.float32)
    out = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"]).run(None, {"features": x})[0]
    assert np.abs(out - clf.predict_proba(x)).max() < 1e-4


def test_engine_spells_words_from_synthetic_keypoints():
    toks = []
    eng = SignEngine("prl", camera=None, on_token=toks.append)
    _spell(eng, ("condor", "sol"))
    assert eng.text() == "cóndor sol"
    assert all(isinstance(t, SignToken) for t in toks)
    letters = "".join(t.value for t in toks if t.kind == "letter")
    assert letters == "condorsol"
    st = eng.state()
    assert set(["hands_visible", "fps", "buffer", "candidates"]) <= set(st)
    assert st["candidates"] and st["candidates"][0]["text"] == "sol"          # swappable last word


def test_engine_accept_backspace_clear():
    eng = SignEngine("ase")
    shapes = {k: np.asarray(v, np.float32) for k, v in eng.speller.clf.prototypes.items()}
    for t, xy, cf in synth.letter_sequence(list("sun"), shapes, rng=np.random.default_rng(0)):
        eng.process_keypoints(t, xy, cf)
    assert eng.state()["buffer"] == "sun"
    eng.backspace()
    assert eng.state()["buffer"] == "su"
    word = eng.accept(0)
    assert word and eng.text() == word and eng.state()["buffer"] == ""
    eng.clear()
    assert eng.text() == ""


def test_engine_thread_with_mock_camera_gives_jpeg(monkeypatch):
    from yq.common import config
    monkeypatch.setattr(config, "MOCK", True)
    statuses = []
    eng = SignEngine("prl", on_status=statuses.append)
    eng.start()
    try:
        for _ in range(100):
            if eng.latest_jpeg():
                break
            time.sleep(0.05)
        jpg = eng.latest_jpeg()
        assert jpg and jpg[:2] == b"\xff\xd8"
        eng.set_mode("words")
        assert eng.state()["mode"] == "words"
        with pytest.raises(ValueError):
            eng.set_mode("sentences")
    finally:
        eng.stop()
    assert statuses and "hands_visible" in statuses[-1]


def test_recording_quality_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    xy, cf = synth.word_sequence(1, 4, rng)
    t = np.arange(len(xy)) / 30.0
    ok, problems, stats = R.check_quality(xy, cf, t, "words", "cóndor")
    assert ok, problems
    bad_cf = cf.copy()
    bad_cf[:, 27:] = 0.0
    ok2, problems2, _ = R.check_quality(xy, bad_cf, t, "words")
    assert not ok2 and any("manos" in p for p in problems2)
    p = R.take_path("prl", "words", "cóndor", "Ignacio", "s1", root=tmp_path)
    R.save_take(p, xy, cf, t, "cóndor", "Ignacio", "s1", "prl", "words", quality=stats)
    d = R.load_take(p)
    assert str(d["label"]) == "cóndor" and d["xy"].shape == xy.shape
    assert p.parent.name == "condor"


def test_word_pipeline_trains_exports_and_recognises(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    sys.path.insert(0, str(V2 / "training" / "sign"))
    import train_words as TW
    import words_data as WD
    items = WD.synthetic(n_classes=4, per_class=12, signers=3)
    classes = sorted({it[1] for it in items})
    monkeypatch.setattr(TW, "device", lambda: "cpu")
    model = TW.fit(items, classes, epochs=6, d=64, layers=1)
    meta = {"classes": classes, "frames": TW.FRAMES, "text": {c: c.lower() for c in classes}}
    onnx_path, parity = TW.export(model, classes, meta, tmp_path, "ils")
    assert parity < 1e-3
    from yq.sign.words import SignSpotter, WordRecognizer
    rec = WordRecognizer.load("ils", path=onnx_path)
    xy, cf = synth.word_sequence(2, 4, np.random.default_rng(99))
    top = rec.predict(xy, cf, k=3)
    assert len(top) == 3 and abs(sum(p for _, p in top)) <= 1.0 + 1e-5
    # the spotter cuts a sign between two rests
    sp = SignSpotter(rest_s=0.3)
    rest_xy, rest_cf = synth.body()
    segs = []
    seq = [(rest_xy, rest_cf)] * 10 + list(zip(xy, cf)) + [(rest_xy, rest_cf)] * 20
    for i, (a, b) in enumerate(seq):
        s = sp.update(i / 30, a, b)
        if s:
            segs.append(s)
    assert len(segs) == 1 and 0.5 < segs[0][3] - segs[0][2] < 2.0
    # the engine in words mode uses the model from MODELS_DIR/sign and offers alternatives
    TW.export(model, classes, dict(meta), modelstore.sign_dir(), "ils")
    assert {d["code"]: d for d in available()}["ils"]["words"] == 4
    toks = []
    eng = SignEngine("ils", on_token=toks.append)
    eng.set_mode("words")
    for i, (a, b) in enumerate(seq):
        eng.process_keypoints(i / 30, a, b)
    st = eng.state()
    assert [t.kind for t in toks] == ["word"] and len(st["candidates"]) == 4
    assert all("_appended" not in c for c in st["candidates"])
    chosen = eng.accept(2)
    assert eng.text() == chosen                    # the alternative replaced (or became) the word


@pytest.mark.heavy
def test_rtm_pipeline_matches_rtmlib(real_models):
    rtmlib = pytest.importorskip("rtmlib")
    import cv2
    from yq.sign import rtm
    path = modelstore.pose_model_path("rtmw-m")
    if path is None:
        pytest.skip("rtmw-m not downloaded")
    img = cv2.imread(str(V2 / "training" / "sign" / "data" / "samples" / "astronaut.png"))
    if img is None:
        img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    box = [10, 15, 360, 500]
    ours = rtm.TopDownPose(path, (192, 256), "cpu")(img, box)
    ref = rtmlib.RTMPose(str(path), model_input_size=(192, 256), backend="onnxruntime", device="cpu")(img, bboxes=[box])
    assert np.abs(ours[0] - ref[0][0]).max() < 0.05 and np.abs(ours[1] - ref[1][0]).max() < 1e-4


@pytest.mark.heavy
def test_pose_estimator_runs_on_image(real_models):
    import cv2
    from yq.sign.pose import PoseEstimator
    if modelstore.pose_model_path("rtmw-m") is None:
        pytest.skip("models not downloaded")
    img = cv2.imread(str(V2 / "training" / "sign" / "data" / "samples" / "astronaut.png"))
    if img is None:
        pytest.skip("sample image missing")
    est = PoseEstimator(model="rtmw-m", backend="cpu")
    pf = est.process(img, 0.0)
    assert pf.person and pf.canon_xy.shape == (69, 2) and pf.scores[0] > 0.5


def test_kaggle_islr_parquet_loader(tmp_path):
    """The asl-signs layout (train.csv + long-format parquet: frame,row_id,type,landmark_index,x,y,z)."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(V2 / "training" / "sign"))
    import words_data as WD
    from yq.sign import skeleton as sk
    xy, cf = synth.word_sequence(0, 4, np.random.default_rng(3))
    W, H = synth.W, synth.H
    rows = []
    for f in range(len(xy)):
        pose = np.full((33, 2), np.nan)
        for c, m in zip(range(13), sk.MP_POSE_TO_CANON_BODY):
            pose[m] = xy[f, c] / [W, H]
        face = np.zeros((468, 2)) + xy[f, sk.C_NOSE] / [W, H]
        parts = {"face": face, "left_hand": xy[f, sk.C_LHAND] / [W, H], "pose": pose,
                 "right_hand": xy[f, sk.C_RHAND] / [W, H]}
        for typ in ("face", "left_hand", "pose", "right_hand"):
            for i, (x, y) in enumerate(parts[typ]):
                rows.append({"frame": 20 + f, "row_id": "%d-%s-%d" % (20 + f, typ, i), "type": typ,
                             "landmark_index": i, "x": x, "y": y, "z": 0.0})
    (tmp_path / "train_landmark_files" / "7").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(tmp_path / "train_landmark_files" / "7" / "1.parquet")
    (tmp_path / "train.csv").write_text("path,participant_id,sequence_id,sign\ntrain_landmark_files/7/1.parquet,7,1,tv\n")
    items = WD.from_kaggle(tmp_path, aspect=W / H)
    assert len(items) == 1 and items[0][1] == "tv" and items[0][2] == "7"
    ref, _ = __import__("yq.sign.features", fromlist=["x"]).normalize_sequence(xy, cf)
    got = items[0][0]
    hands = slice(sk.C_LHAND.start * 2, sk.N_CANON * 2)          # hands + body agree (face was faked)
    assert np.allclose(got[:, hands], ref[:, hands], atol=1e-3)


def test_replay_source_feeds_engine(tmp_path):
    """A recorded take (as written by the recorder) replays through the engine."""
    eng = SignEngine("prl")
    shapes = {k: np.asarray(v, np.float32) for k, v in eng.speller.clf.prototypes.items()}
    frames = list(synth.letter_sequence(list("sol"), shapes, rng=np.random.default_rng(5)))
    t = np.array([f[0] for f in frames])
    xy = np.stack([f[1] for f in frames])
    cf = np.stack([f[2] for f in frames])
    p = R.take_path("prl", "letters", "sol", "test", "s1", root=tmp_path)
    R.save_take(p, xy, cf, t, "sol", "test", "s1", "prl", "letters")
    src = sources.ReplaySource([p], realtime=False)
    while True:
        g = src.next()
        if g is None:
            break
        eng.process_keypoints(*g)
    assert eng.state()["buffer"] == "sol"


@pytest.mark.parametrize("lang", ["ase", "ils", "prl"])
def test_tapping_a_chip_commits_that_chip(lang):
    """accept(i) must commit exactly state()['candidates'][i] (it used to re-rank with final=True:
    tapping 'help' committed 'hell')."""
    eng = SignEngine(lang)
    shapes = {k: np.asarray(v, np.float32) for k, v in eng.speller.clf.prototypes.items()}
    for t, xy, cf in synth.letter_sequence(list("hel" if lang != "prl" else "mon"), shapes,
                                           rng=np.random.default_rng(1)):
        eng.process_keypoints(t, xy, cf)
    shown = [c["text"] for c in eng.state()["candidates"]]
    assert len(shown) >= 2, shown
    i = len(shown) - 1                                        # the last chip, the one most likely to differ
    assert eng.accept(i) == shown[i]
    assert eng.text().split()[-1] == shown[i]
