from __future__ import annotations

import numpy as np

from yq.sign import lexicon as lx
from yq.sign.filters import OneEuroFilter
from yq.sign.letters import LetterSegmenter, MotionTracker

CLS = ["a", "b", "l", "s"]


def onehot(letter, p=0.9):
    v = np.full(len(CLS), (1 - p) / (len(CLS) - 1), np.float32)
    v[CLS.index(letter)] = p
    return v


def run(seg, script, fps=30):
    """script: list of (letter|None, seconds, still, wrist_xy). Returns committed letters."""
    out, t = [], 0.0
    for letter, secs, still, wrist in script:
        for _ in range(int(secs * fps)):
            c = seg.update(t, None if letter is None else onehot(letter), CLS, still=still,
                           wrist=None if wrist is None else np.array(wrist, np.float32))
            if c:
                out.append(c.letter)
            t += 1 / fps
    return out


def test_hold_commits_once_and_transitions_do_not():
    seg = LetterSegmenter(hold_s=0.3)
    assert run(seg, [("a", 1.0, True, (0, 0))]) == ["a"]              # long hold = one letter
    seg.reset()
    assert run(seg, [("a", 0.1, True, (0, 0)), ("b", 0.1, True, (0, 0)), ("s", 0.1, True, (0, 0))]) == []
    seg.reset()
    assert run(seg, [("a", 1.0, False, (0, 0))]) == []                 # moving hand never commits


def test_double_letter_needs_a_bounce():
    seg = LetterSegmenter(hold_s=0.3, rearm_motion=0.35)
    same_place = [("l", 0.6, True, (0, 0)), ("l", 0.6, True, (0.05, 0))]
    assert run(seg, same_place) == ["l"]
    seg.reset()
    bounce = [("l", 0.6, True, (0, 0)), ("l", 0.2, False, (0.5, 0.2)), ("l", 0.6, True, (0.1, 0))]
    assert run(seg, bounce) == ["l", "l"]
    seg.reset()
    hand_left = [("l", 0.6, True, (0, 0)), (None, 0.3, True, None), ("l", 0.6, True, (0, 0))]
    assert run(seg, hand_left) == ["l", "l"]


def test_letter_change_commits_next_letter():
    seg = LetterSegmenter(hold_s=0.3)
    assert run(seg, [("a", 0.6, True, (0, 0)), ("b", 0.6, True, (0, 0)), ("a", 0.6, True, (0, 0))]) == ["a", "b", "a"]


def _hand(offset, shape_top):
    h = np.zeros((21, 2), np.float32)
    h[[5, 9, 13, 17]] = [[-0.3, -1], [0, -1], [0.3, -1], [0.5, -0.9]]
    h[1:5] = [[-0.3, -0.3], [-0.6, -0.5], [-0.8, -0.7], [-0.9, -0.9]]
    for f, b in ((6, 5), (10, 9), (14, 13), (18, 17)):
        h[f:f + 3] = h[b] + np.array([[0, -0.4], [0, -0.7], [0, -1.0]])
    return h * 40 + np.asarray(offset) * 40, shape_top


def test_motion_tracker_z_and_not_shape_change():
    mt = MotionTracker("ase")
    pts = np.array([[0, 0], [1.6, 0], [0, 1.4], [1.6, 1.4]])
    t = 0.0
    found = None
    for k in range(30):                       # Z path over 1 s with the Z hand shape
        u = k / 29 * 3
        seg = min(int(u), 2)
        pos = pts[seg] + (pts[seg + 1] - pts[seg]) * (u - seg)
        h, top = _hand(pos, "z")
        mt.push(t, h, False, top)
        found = found or mt.detect(top, t)
        t += 1 / 30
    assert found == "z"
    mt.reset()
    for k in range(40):                       # finger changes without moving the wrist
        h, _ = _hand((0, 0), "d")
        h[8] += np.array([np.sin(k), np.cos(k)]) * 40
        mt.push(k / 30, h, False, "d")
        assert mt.detect("d", k / 30) is None


def test_motion_tracker_enie_side_to_side_lsp_only():
    for lang, want in (("prl", "ñ"), ("ase", None)):
        mt = MotionTracker(lang)
        found = None
        for k in range(30):
            h, top = _hand((0.9 * np.sin(2 * np.pi * 1.5 * k / 29), 0), "n")
            mt.push(k / 30, h, False, top)
            found = found or mt.detect(top, k / 30)
        assert found == want


def test_one_euro_smooths_jitter_and_follows_moves():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    still = [f(np.array([[100.0, 100.0]]) + rng.normal(0, 2, (1, 2)), k / 30, scale=200.0) for k in range(60)]
    assert np.std(np.array(still)[30:, 0, 0]) < 1.0                  # raw std is 2 px
    f.reset()
    out = [f(np.array([[100.0 + 20 * k, 100.0]]), k / 30, scale=200.0) for k in range(30)]
    assert abs(out[-1][0, 0] - (100 + 20 * 29)) < 60                 # small lag at 600 px/s
    f.reset()
    f(np.array([[0.0, 0.0]]), 0.0, conf=np.array([0.9]))
    held = f(np.array([[500.0, 500.0]]), 1 / 30, conf=np.array([0.05]))   # low confidence ignored
    assert np.allclose(held, 0.0)


def test_lexicon_completion():
    lex = lx.Lexicon({"cóndor": 50, "condición": 400, "sol": 900, "hello": 5000, "helo": 20, "help": 3000,
                      "sach'a": 30, "llama": 300, "lama": 200})
    obs = [{"c": .9}, {"o": .9}, {"n": .7, "m": .2}, {"d": .9}, {"o": .9}, {"r": .8, "u": .1}]
    assert lx.complete(lex, obs, final=True)[0][0] == "cóndor"
    assert lx.complete(lex, [{"h": 1}, {"e": 1}, {"l": 1}, {"o": 1}], final=True)[0][0] == "hello"   # missed LL
    assert lx.complete(lex, [{c: 1.0} for c in "sacha"], final=True)[0][0] == "sach'a"
    assert lx.fold("Cóndor") == "condor" and lx.fold("niño") == "niño"
    top = lx.complete(lex, [{"x": 1}, {"q": 1}], final=True)
    assert top[0][0] == "xq"                                          # names outside the lexicon survive
    assert abs(sum(p for _, p in top) - 1) < 1e-6


def test_shipped_lexicons_load_and_are_fast():
    import time
    lex = lx.get("prl")
    assert lex is not None and len(lex) > 20000 and "cóndor" in lex and "kuntur" in lex
    t0 = time.perf_counter()
    for _ in range(10):
        lx.complete(lex, [{"m": .6, "n": .4}, {"o": 1}, {"n": 1}, {"t": 1}, {"a": 1}])
    assert (time.perf_counter() - t0) / 10 < 0.05
    assert "montaña" in [w for w, _ in lx.complete(lex, [{c: 1.0} for c in "montana"], final=True)]


def test_one_euro_with_default_settings_scaled_by_box():
    from yq.sign import settings
    rng = np.random.default_rng(1)
    f = OneEuroFilter(settings.SIGN_EURO_MIN_CUTOFF, settings.SIGN_EURO_BETA, settings.SIGN_EURO_D_CUTOFF)
    box = 600.0
    still = [f(np.array([[300.0, 300.0]]) + rng.normal(0, 1.5, (1, 2)), k / 30, scale=box) for k in range(90)]
    assert np.std(np.array(still)[45:, 0, 0]) < 0.75                  # jitter at least halved
    f.reset()
    moving = [f(np.array([[300.0 + 10 * k, 300.0]]), k / 30, scale=box) for k in range(30)]   # 300 px/s
    assert abs(moving[-1][0, 0] - (300 + 290)) < 15                   # < 1.5 frames of lag
