"""The real serial driver (pyserial + threads) against the simulator on a pty."""
import time

import pytest

from yq.box.protocol import BoxError, BoxTimeout, BoxUnavailable


def test_connect_info_status(simbox):
    box = simbox()
    assert box.info_cache["fw"] == "yq-box" and box.info_cache["steps_per_rev"] == 44800
    st = box.status()
    assert st["doors_closed"] and st["motor"]["tmc_ok"] and st["scale"]["calibrated"]
    assert box.ping()["pong"] is True


def test_rotation_timing_and_move_done(simbox):
    box = simbox(time_scale=5.0)
    t0 = time.time()
    done = box.rotate_to(90.0)
    real = time.time() - t0
    # 90 deg at 30 deg/s, 45 deg/s^2: 90/30 + 30/45 = 3.67 s of box time -> /5 real
    assert done["deg"] == pytest.approx(90.0, abs=0.01) and not done["stopped"]
    assert 0.6 < real < 1.6
    done = box.rotate_by(-30.0)
    assert done["deg"] == pytest.approx(60.0, abs=0.01)
    assert box.status()["motor"]["steps"] == round(60 * 44800 / 360)


def test_rotate_by_is_sent_as_absolute_target(simbox):
    box = simbox()
    for _ in range(36):
        box.rotate_by(10.0)
    assert box.status()["motor"]["deg"] == pytest.approx(360.0, abs=0.005)   # no accumulated rounding


def test_errors_are_raised_with_codes(simbox):
    box = simbox()
    with pytest.raises(BoxError) as e:
        box.request("light", ch="laser", level=1.0)
    assert e.value.code == "bad_param"
    box.rotate_to(10, wait=False)
    with pytest.raises(BoxError) as e:
        box.zero()
    assert e.value.code == "moving"


def test_timeout_and_retry_of_idempotent_command(simbox):
    box = simbox()
    link = box.link
    link.connected = False                        # cable "cut": bytes vanish
    t0 = time.time()
    with pytest.raises(BoxTimeout):
        box.request("status", timeout=0.2, retries=1)
    assert 0.35 < time.time() - t0 < 2.0          # two attempts
    link.connected = True
    assert box.status()["ok"]


def test_events_reach_listeners(simbox):
    box = simbox()
    seen = []
    box.add_listener(seen.append)
    box.sim.set_door("shutter", False)
    ev = box.wait_event("door", lambda m: m["door"] == "shutter", timeout=2)
    assert ev["closed"] is False and any(m.get("event") == "door" for m in seen)
    box.sim.set_door("shutter", True)
    box.wait_event("door", lambda m: m["closed"], timeout=2)
    assert box.doors_closed()


def test_board_reset_is_detected(simbox):
    box = simbox()
    box.rotate_to(45)
    box.link.reset_board()
    box.wait_event("boot", timeout=2)
    assert box.resets == 1
    assert box.status()["motor"]["deg"] == 0.0    # position is lost after a reset


def test_reconnect_after_port_error(simbox):
    box = simbox()
    box._drop("simulated unplug")                 # what a read error does
    with pytest.raises(BoxUnavailable):
        box.request("ping", retries=0)
    t0 = time.time()
    while not box.connected and time.time() - t0 < 5:
        time.sleep(0.05)
    assert box.connected and box.ping()["pong"]
