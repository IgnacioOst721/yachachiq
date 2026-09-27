"""Scale: the stability rule and the tare / calibrate / weigh flow on the simulator."""
import random

import pytest

from yq.box.protocol import BoxError
from yq.box.scalemath import find_stable_window, is_stable, raw_to_grams, window_stats


def test_window_stats_and_stability():
    rng = random.Random(1)
    quiet = [500 + rng.gauss(0, 0.1) for _ in range(20)]
    assert is_stable(quiet, 0.5)
    st = window_stats(quiet)
    assert st.mean == pytest.approx(500, abs=0.1) and st.sigma < 0.2
    noisy = [500 + rng.gauss(0, 2.0) for _ in range(20)]      # fan on / vibration
    assert not is_stable(noisy, 0.5)


def test_slow_drift_is_not_stable_even_with_small_sigma():
    creep = [500 + 0.08 * i for i in range(20)]               # 1.6 g creep over 2 s, sigma 0.47
    st = window_stats(creep)
    assert st.sigma < 0.5 and st.drift > 0.5
    assert not is_stable(creep, 0.5)


def test_find_stable_window_after_settling():
    import math
    rng = random.Random(2)
    settling = [812 * (1 - math.exp(-i / 5.0)) + rng.gauss(0, 0.1) for i in range(80)]
    end = find_stable_window(settling, 20, 0.5)
    assert end is not None and settling[end - 1] == pytest.approx(812, abs=1.0)
    assert raw_to_grams(-391145 + -412.3 * 100, -391145, -412.3) == pytest.approx(100)


def test_full_calibration_flow_on_uncalibrated_scale(simbox):
    box = simbox(time_scale=10.0, calibrated=False)
    with pytest.raises(BoxError) as e:
        box.weigh()
    assert e.value.code == "not_tared"
    box.tare()                                                # empty platter
    with pytest.raises(BoxError) as e:
        box.weigh()
    assert e.value.code == "not_calibrated"
    box.sim.place_object(500.0)                               # known calibration mass
    cal = box.calibrate_scale(500.0)
    assert cal["factor"] == pytest.approx(-412.3, rel=0.01)
    box.sim.place_object(812.5)
    w = box.weigh()
    assert w["stable"] and w["grams"] == pytest.approx(812.5, abs=1.0)
    assert len(w["samples"]) == 20 and w["sigma_g"] < 0.5
    assert box.sim.nvs["scale"]["factor"] == pytest.approx(cal["factor"])   # persisted like NVS


def test_calibration_survives_a_reboot(simbox):
    box = simbox(time_scale=10.0, calibrated=False)
    box.tare()
    box.sim.place_object(1000.0)
    box.calibrate_scale(1000.0)
    box.link.reset_board()
    box.wait_event("boot", timeout=2)
    st = box.status()["scale"]
    assert st["tared"] and st["calibrated"]
    assert box.weigh()["grams"] == pytest.approx(1000.0, abs=1.0)


def test_weighing_is_refused_while_turning_and_fan_is_held(simbox):
    box = simbox(time_scale=2.0)
    box.rotate_to(90, wait=False)
    with pytest.raises(BoxError) as e:
        box.weigh()
    assert e.value.code == "moving"
    box.wait_event("move_done", timeout=10)
    box.light("fan", 1.0)
    w = box.weigh(n=10)
    assert w["stable"]                                        # the fan was paused for the measurement
    assert box.sim.channels["fan"].level == 1.0 and not box.sim.channels["fan"].held


def test_unstable_reading_times_out_honestly(simbox):
    box = simbox(time_scale=10.0)
    box.sim.scale.noise_g = 3.0                               # e.g. someone leaning on the table
    w = box.weigh(n=20, timeout_ms=3000)
    assert w["stable"] is False and w["sigma_g"] > 0.5
    with pytest.raises(BoxError) as e:
        box.tare(timeout_ms=2000)
    assert e.value.code == "scale_timeout"
