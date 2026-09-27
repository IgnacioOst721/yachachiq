"""Thermal: Y16 decoding, frozen-frame detection and the mock thermography physics."""
import numpy as np
import pytest

from yq.box import thermal as T


def test_centikelvin_conversion_matches_groupgets_ktoc():
    # uvc-radiometry.py: ktoc(val) = (val - 27315) / 100.0
    assert T.centikelvin_to_c(29315) == pytest.approx(20.0, abs=1e-4)
    assert T.centikelvin_to_c(27315) == pytest.approx(0.0, abs=1e-4)
    assert T.c_to_centikelvin(36.6) == 30975
    raw = np.array([[30315, 29815]], np.uint16)
    assert np.allclose(T.centikelvin_to_c(raw), [[30.0, 25.0]], atol=1e-4)


def test_to_frame_u16_shapes():
    img = np.full((120, 160), 29500, np.uint16)
    assert T.to_frame_u16(img).shape == (120, 160)
    tele = np.full((122, 160), 29500, np.uint16)          # 2 telemetry rows at the bottom
    assert T.to_frame_u16(tele).shape == (120, 160)
    buf = img.astype("<u2").tobytes()
    assert np.array_equal(T.to_frame_u16(np.frombuffer(buf, np.uint8)), img)
    with pytest.raises(ValueError):
        T.to_frame_u16(np.zeros((120, 160, 3), np.uint8))  # someone left CONVERT_RGB on
    with pytest.raises(ValueError):
        T.to_frame_u16(np.zeros((60, 80), np.uint16))


def test_tlinear_check():
    assert T.looks_like_tlinear(np.full((120, 160), 29515, np.uint16))
    assert not T.looks_like_tlinear(np.full((120, 160), 8200, np.uint16))   # raw counts, TLinear off


def test_detect_frozen():
    rng = np.random.default_rng(0)
    frames = [rng.integers(29000, 29100, (120, 160)).astype(np.uint16) for _ in range(6)]
    frames[3] = frames[2].copy()
    frames[4] = frames[2].copy()
    fz = T.detect_frozen(np.stack(frames))
    assert fz.tolist() == [False, False, False, True, True, False]


def test_mock_thermography_shows_the_defect():
    from yq.box.mockthermal import MockThermal
    state = {"lights": {"halogen": 0.0}, "object_g": 800.0, "time_scale": 1e6}
    cam = MockThermal(lambda: state).open()

    def on_frame(t, _):
        state["lights"]["halogen"] = 1.0 if 5.0 <= t < 20.0 else 0.0

    rec = T.record(cam, 60.0, on_frame=on_frame)
    f, t = rec["frames_c"], rec["times"]
    assert f.dtype == np.float32 and f.shape[1:] == (120, 160)
    assert np.all(np.diff(t) > 0) and t[-1] >= 60.0
    assert len(t) == pytest.approx(60 * 8.7, abs=3)
    end_heat = int(np.searchsorted(t, 20.0)) - 1
    d, n = cam.defect, cam.mask & ~cam.defect
    base = f[:40]
    assert abs(float(np.median(base)) - 22.0) < 0.2           # ambient before heating
    rise_n = f[end_heat][n].mean() - base[:, n].mean()
    rise_d = f[end_heat][d].mean() - base[:, d].mean()
    assert 1.5 < rise_n < 10 and rise_d > rise_n + 1.5        # the defect heats more
    assert f[-1][n].mean() < f[end_heat][n].mean() - 1.0      # and everything cools afterwards
    assert rec["frozen"].sum() >= 2                           # one FFC freeze flagged
