"""Firmware safety logic (as simulated): interlock, on-time limits, cool-downs, heartbeat."""
import time

import pytest

from yq.box.protocol import BoxError


def test_uv_and_halogen_refuse_with_a_door_open(simbox):
    box = simbox(doors_closed=False)
    for ch in ("uv", "halogen"):
        with pytest.raises(BoxError) as e:
            box.light(ch, 1.0)
        assert e.value.code == "interlock"
    box.light("rake1", 1.0, max_ms=500)          # visible LEDs are allowed with the door open
    box.sim.set_door("front", True)
    box.sim.set_door("shutter", True)
    time.sleep(0.1)                               # 50 ms debounce before "closed" counts
    assert box.light("uv", 1.0, max_ms=1000)["level"] == 1.0


@pytest.mark.parametrize("door", ["front", "shutter"])
def test_door_opening_cuts_uv_and_halogen_within_10_ms(simbox, door):
    box = simbox(time_scale=1.0)
    box.light("uv", 1.0, max_ms=5000)
    box.light("halogen", 1.0, max_ms=5000)
    sim = box.sim
    t_open = time.monotonic()
    sim.set_door(door, False)                     # like the reed ISR
    cut_ms = (time.monotonic() - t_open) * 1000
    assert sim.channels["uv"].level == 0 and sim.channels["halogen"].level == 0
    assert cut_ms < 10.0
    ev = box.wait_event("interlock", timeout=1.0)
    assert set(ev["cut"]) == {"uv", "halogen"}
    assert not box.status()["doors_closed"]


def test_broken_reed_wire_counts_as_open(simbox):
    box = simbox()
    box.sim.set_door("front", False)              # broken wire = HIGH through the pull-up = open
    with pytest.raises(BoxError):
        box.light("halogen", 1.0)


def test_max_on_time_switches_off_and_reports(simbox):
    box = simbox(time_scale=10.0)
    r = box.light("rake2", 1.0)                   # default rake limit 20 s of box time = 2 s real
    assert 1900 <= r["off_in_ms"] <= 2000
    ev = box.wait_event("light_off", lambda m: m["ch"] == "rake2", timeout=4.0)
    assert ev["reason"] == "max_on"
    assert box.status()["lights"]["rake2"]["level"] == 0


def test_cooldown_blocks_reuse_then_allows(simbox):
    box = simbox(time_scale=1.0)
    box.light("rake4", 1.0)
    time.sleep(0.3)
    box.light("rake4", 0)
    with pytest.raises(BoxError) as e:
        box.light("rake4", 1.0)
    assert e.value.code == "cooldown" and 300 <= e.value.reply["wait_ms"] <= 700   # 2 x 0.3 s
    time.sleep(e.value.reply["wait_ms"] / 1000 + 0.05)
    assert box.light("rake4", 1.0, max_ms=200)["level"] == 1.0


def test_halogen_limit_and_triple_cooldown(simbox):
    box = simbox(time_scale=100.0)
    box.light("halogen", 1.0)                     # 45 s max -> 0.45 s real
    box.wait_event("light_off", lambda m: m["ch"] == "halogen", timeout=2)
    with pytest.raises(BoxError) as e:
        box.light("halogen", 1.0)
    assert e.value.code == "cooldown" and 1200 <= e.value.reply["wait_ms"] <= 1400   # 3 x 45 s / 100


def test_only_one_raking_led_at_a_time(simbox):
    box = simbox()
    box.light("rake1", 1.0)
    box.light("rake5", 1.0)
    ev = box.wait_event("light_off", lambda m: m["ch"] == "rake1", timeout=1)
    assert ev["reason"] == "exclusive"
    on = [k for k, v in box.status()["lights"].items() if v["level"] > 0]
    assert on == ["rake5"]


def test_config_cannot_exceed_hard_caps(simbox):
    box = simbox()
    cfg = box.config_set(max_on_ms={"halogen": 600000, "rake": 5000}, cool_factor={"halogen": 0.5})
    assert cfg["max_on_ms"]["halogen"] == 60000 and cfg["max_on_ms"]["rake"] == 5000
    assert cfg["cool_factor"]["halogen"] == 2.0
    assert box.config_set(hb_timeout_ms=60000)["hb_timeout_ms"] == 10000


def test_heartbeat_loss_turns_everything_off(simbox):
    box = simbox(heartbeat=False)
    box.config_set(hb_timeout_ms=600)
    box.light("fan", 1.0)
    box.light("cob", 1.0)
    box.motor(True)
    ev = box.wait_event("watchdog", timeout=3.0)
    assert ev["reason"] == "heartbeat"
    st = box.status()                              # this request re-arms the watchdog
    assert all(v["level"] == 0 for v in st["lights"].values())
    assert st["motor"]["enabled"] is False and st["hb_lost"] is False


def test_heartbeat_thread_keeps_the_box_alive(simbox, monkeypatch):
    from yq.box import settings as S
    monkeypatch.setattr(S, "HEARTBEAT_S", 0.2)
    box = simbox(heartbeat=True)
    box.config_set(hb_timeout_ms=600)
    box.light("fan", 1.0)
    time.sleep(1.5)
    assert box.sim.channels["fan"].level == 1.0
    assert not box.events("watchdog")


def test_door_open_stops_the_platter(simbox):
    box = simbox(time_scale=1.0)
    r = box.rotate_to(180, wait=False)
    time.sleep(0.5)
    box.sim.set_door("front", False)
    done = box.wait_move(r["id"], 5.0)
    assert done["stopped"] is True and done["deg"] < 60
    with pytest.raises(BoxError) as e:           # no new motion while the door is open
        box.rotate_to(90)
    assert e.value.code == "interlock"
    box.config_set(door_stops_motor=False)       # bench testing with the door open
    assert box.rotate_to(90)["deg"] == pytest.approx(90, abs=0.01)
