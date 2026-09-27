"""Vectorizer accuracy on synthetic line art (known centre lines)."""
from __future__ import annotations

import numpy as np
import pytest

from yq.art import geom, text, vectorize

W, H = 1000, 1400


def _trace(src, width_px, **kw):
    img = geom.render(src, W, H, width_px=width_px)
    kw.setdefault("drop_frame", False)
    return vectorize.trace(img, work_px=1400, px_per_mm=7.4, **kw)


def _dist_stats(A, B):
    """(max, 95th percentile) of the symmetric point-to-polyline distances."""
    pa = np.concatenate([geom.resample(s, 0.5) for s in A])
    pb = np.concatenate([geom.resample(s, 0.5) for s in B])
    d = np.concatenate([geom.point_segment_dist(pa, B), geom.point_segment_dist(pb, A)])
    return float(d.max()), float(np.percentile(d, 95))


def _shapes():
    t = np.linspace(0, 2 * np.pi, 400)
    x = np.linspace(100, 900, 400)
    return {
        "line": [np.array([[100.0, 100.0], [900.0, 160.0]])],
        "circle": [np.stack([500 + 200 * np.cos(t), 600 + 200 * np.sin(t)], 1)],
        "curve": [np.stack([x, 1100 + 80 * np.sin(x / 60)], 1)],
        "spiral": [np.stack([500 + (40 + 30 * t) * np.cos(t * 1.5), 700 + (40 + 30 * t) * np.sin(t * 1.5)], 1)],
    }


@pytest.mark.parametrize("name", ["line", "circle", "curve", "spiral"])
@pytest.mark.parametrize("width_px", [3, 5, 8])
def test_hausdorff_simple_shapes(name, width_px):
    src = _shapes()[name]
    tr = _trace(src, width_px)
    assert len(tr.strokes) == 1, tr.info
    hd, p95 = _dist_stats(src, tr.strokes)
    # Hausdorff (worst point, usually a line end) <= 2 px; 95 % of the drawing within 1 px
    assert hd <= 2.0 and p95 <= 1.0, "%s w=%d: Hausdorff %.2f px, p95 %.2f px" % (name, width_px, hd, p95)


@pytest.mark.parametrize("width_px", [3, 5, 7])
def test_hausdorff_letters(width_px):
    """Hershey letters (with accents) as source geometry: many junctions, corners, short stubs.
    Measured: mean ~0.5 px, 95 % within ~1.3 px; the worst point (a 1-mm stub where the stem of
    an 'a' leaves its bowl) can be ~8 px off, so the maximum is only bounded loosely."""
    src = text.render_line("Ñandú ABRKX", 0, 0, 80.0, "futural")
    src = [s + np.array([60.0, 400.0]) for s in src]
    b = geom.bounds(src)
    assert b[2] < W and b[3] < H
    img = geom.render(src, W, H, width_px=width_px)
    tr = vectorize.trace(img, work_px=1400, px_per_mm=4.0, drop_frame=False)
    pa = np.concatenate([geom.resample(s, 0.5) for s in src])
    pb = np.concatenate([geom.resample(s, 0.5) for s in tr.strokes])
    d = np.concatenate([geom.point_segment_dist(pa, tr.strokes), geom.point_segment_dist(pb, src)])
    assert d.mean() <= 0.7 and np.percentile(d, 95) <= 1.6 and d.max() <= 9.0, \
        (d.mean(), np.percentile(d, 95), d.max())
    # merging through junctions: fewer strokes than the font itself uses
    assert len(tr.strokes) <= len(src)


def test_junctions_are_merged_straight_through():
    src = [np.array([[150.0, 300.0], [450.0, 300.0]]), np.array([[300.0, 300.0], [300.0, 500.0]]),   # T
           np.array([[600.0, 250.0], [850.0, 480.0]]), np.array([[850.0, 250.0], [600.0, 480.0]]),   # X
           np.array([[200.0, 900.0], [600.0, 900.0]]), np.array([[400.0, 700.0], [400.0, 1100.0]])]  # +
    for w in (3, 6):
        tr = _trace(src, w)
        assert len(tr.strokes) == 6, (w, tr.info)
        # every result stroke is straight (one line through the crossing, not a bent piece)
        for s in tr.strokes:
            chord = np.hypot(*(s[-1] - s[0]))
            assert geom.length(s) < chord * 1.03
        assert geom.hausdorff(src, tr.strokes) <= 2.0


def test_filled_region_becomes_outline_and_hatching():
    import cv2
    img = np.full((H, W), 255, np.uint8)
    cv2.circle(img, (500, 700), 180, 0, -1)                      # a solid black disk
    cv2.line(img, (100, 200), (900, 200), 0, 4)
    tr = vectorize.trace(img, work_px=1400, px_per_mm=7.4, drop_frame=False, fill_style="hatch")
    assert tr.info["fill_fraction"] > 0.05
    t = np.linspace(0, 2 * np.pi, 300)
    ring = [np.stack([500 + 180 * np.cos(t), 700 + 180 * np.sin(t)], 1)]
    # some stroke follows the disk outline closely
    outline = [s for s in tr.strokes if geom.length(s) > 900]
    assert outline and geom.hausdorff(ring, outline[:1]) < 3.0
    # and hatching lines lie inside the disk
    inside = [s for s in tr.strokes if np.all(np.hypot(s[:, 0] - 500, s[:, 1] - 700) < 181) and geom.length(s) > 50]
    assert len(inside) >= 1
    # hatch + cross-hatch (black) linked into serpentines: a few dozen strokes, not hundreds
    assert len(tr.strokes) < 80


def test_frame_is_removed():
    import cv2
    img = np.full((H, W), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (W - 21, H - 21), 0, 6)
    cv2.circle(img, (500, 700), 150, 0, 4)
    tr = vectorize.trace(img, work_px=1400, px_per_mm=7.4, drop_frame=True)
    assert len(tr.strokes) == 1
    b = geom.bounds(tr.strokes)
    assert b[0] > 300 and b[2] < 700


def test_small_isolated_details_survive_and_specks_do_not():
    import cv2
    img = np.full((H, W), 255, np.uint8)
    cv2.circle(img, (500, 700), 250, 0, 4)             # a face
    cv2.circle(img, (420, 640), 7, 0, -1)               # eyes: small dots, must stay
    cv2.circle(img, (580, 640), 7, 0, -1)
    img[100, 100] = 0                                   # single-pixel noise, must go
    img[1200:1202, 800:802] = 0
    tr = vectorize.trace(img, work_px=1400, px_per_mm=7.4, drop_frame=False)
    near = lambda x, y: [s for s in tr.strokes if np.min(np.hypot(s[:, 0] - x, s[:, 1] - y)) < 10]
    assert near(420, 640) and near(580, 640)
    assert not near(100, 100) and not near(800, 1200)


def test_white_on_black_is_inverted():
    import cv2
    img = np.zeros((H, W), np.uint8)
    cv2.line(img, (100, 700), (900, 700), 255, 5)
    tr = vectorize.trace(img, work_px=1400, px_per_mm=7.4, drop_frame=False)
    assert len(tr.strokes) == 1
    assert geom.hausdorff([np.array([[100.0, 700.0], [900.0, 700.0]])], tr.strokes) < 2.0


def test_numpy_thinning_matches_opencv():
    import cv2
    from yq.art import skeleton
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, (100, 100), 60, 1, 6)
    cv2.line(img, (20, 20), (180, 60), 1, 5)
    a = skeleton.minimal_skeleton(skeleton._guo_hall(img))
    b = skeleton.thin(img)
    ya, xa = np.nonzero(a)
    pa = np.stack([xa, ya], 1).astype(float)
    yb, xb = np.nonzero(b)
    pb = np.stack([xb, yb], 1).astype(float)
    # both skeletons describe the same centre lines (within 1.5 px)
    d = np.min(np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1]), axis=1)
    assert d.max() <= 1.5
