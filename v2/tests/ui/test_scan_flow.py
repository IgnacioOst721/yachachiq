"""Object analysis flow end-to-end through the real server, UI mocks."""
from __future__ import annotations

import json
import struct

from yq.server import mocks_art_box
from tests.ui.conftest import wait_for


def _start_scan(client, events, profile="quick"):
    assert client.post("/api/start", json={"flow": "scan"}).status_code == 200
    intro = events.screen("intro")
    assert [p["id"] for p in intro["data"]["profiles"]] == ["quick", "standard", "detailed"]
    events.act("choose_profile", profile=profile)
    events.until(lambda m: m["type"] == "screen_update" and m["data"].get("selected") == profile, what="profile")
    events.act("start")
    events.screen("preflight")


def test_scan_full_results_and_viewer_files(client, events):
    _start_scan(client, events)
    s = events.screen("scanning")
    assert s["data"]["weight_g"] == 312.4 and s["data"]["profile"] == "quick"
    res = events.screen("results", timeout=30)["data"]["result"]
    photos = [m for m in events.all if m["type"] == "scan_photo"]
    ring = [m for m in photos if m["kind"] == "photogrammetry"]
    assert len(ring) == 48 and {m["camera"] for m in ring} == {"A", "B"}          # 24 stops x 2 cameras
    assert ring[0]["url"].startswith("/files/scans/") and ring[0]["angle"] == 0.0
    assert client.get(ring[0]["url"]).content[:2] == b"\xff\xd8"
    assert {m["kind"] for m in photos} >= {"rti", "uv"}
    live = [m for m in events.all if m["type"] == "scan_live"]
    assert any(m["detail"].get("light") == "led3" for m in live)
    th = [m for m in live if m["detail"].get("thermal_preview_url")]
    assert th and client.get(th[-1]["detail"]["thermal_preview_url"]).status_code == 200
    assert res["mock"] is True and res["ok"] is True
    assert {m["name"] for m in res["measurements"]} >= {"mass", "height", "width", "depth"}
    assert res["identification"]["culture"] == "Moche" and res["identification"]["alternatives"]
    assert {f["analysis"] for f in res["findings"]} == {"uv", "thermal", "rti"}
    v = res["viewers"]
    glb = client.get(v["model"])
    assert glb.headers["content-type"] == "model/gltf-binary"
    magic, version, length = struct.unpack("<III", glb.content[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(glb.content)
    ptm = client.get(v["rti"]).json()
    assert ptm["format"] == "yq-ptm-lrgb-v1" and len(ptm["scale"]) == 6
    base = v["rti"].rsplit("/", 1)[0]
    for name in ptm["coeff_images"] + [ptm["albedo"]]:
        assert client.get(base + "/" + name).status_code == 200
    for url in (v["uv"]["uv"], v["uv"]["overlay"], v["thermal"]["anomaly"], v["thermal"]["max"]):
        assert client.get(url).status_code == 200
    f = [x for x in res["findings"] if x["analysis"] == "uv"][0]
    assert f["image_url"] == v["uv"]["overlay"]
    events.act("new_scan")                        # "analizar otro objeto" starts a fresh scan flow
    events.screen("intro")


def test_preflight_problems_then_retry(client, events, monkeypatch):
    monkeypatch.setattr(mocks_art_box.MockBox, "problems", ["La puerta está abierta. Ciérrala, por favor."])
    _start_scan(client, events)
    fail = events.screen("preflight_fail")
    assert fail["data"]["problems"] == ["La puerta está abierta. Ciérrala, por favor."]
    monkeypatch.setattr(mocks_art_box.MockBox, "problems", [])
    events.act("retry")
    events.screen("preflight")
    events.screen("scanning")
    client.post("/api/cancel")
    events.screen("home", timeout=20)


def test_stop_scan_turns_box_off_and_frees_it(client, events):
    from yq.common import config
    _start_scan(client, events)
    scan_id = events.screen("scanning")["data"]["scan_id"]
    events.until(lambda m: m["type"] == "scan_photo", what="first photo")
    events.act("stop_scan")
    events.screen("home", timeout=20)
    end = [m for m in events.all if m["type"] == "flow_end"][-1]
    assert end["outcome"] == "stopped"
    assert not client.kiosk.sub.box.lock.locked()          # run_scan returned: lights/heater off
    assert not (config.SCANS_DIR / scan_id / "result.json").exists()
    _start_scan(client, events)                    # the box is free again
    events.screen("scanning")
    client.post("/api/cancel")
    events.screen("home", timeout=20)


def test_scan_failure_shows_kind_error_and_retry_works(client, events, monkeypatch):
    monkeypatch.setattr(mocks_art_box.MockBox, "fail_at", "thermal")
    _start_scan(client, events)
    err = events.screen("error", timeout=30)
    assert err["data"]["key"] == "err_generic" and err["data"]["failed"] == "scanning"
    assert "RuntimeError" in err["data"]["detail"]
    events.act("retry")                            # retry_map: scanning -> preflight
    events.screen("preflight")
    events.screen("results", timeout=30)


def test_results_written_to_scan_folder(client, events):
    from yq.common import config
    _start_scan(client, events)
    res = events.screen("results", timeout=30)["data"]["result"]
    folder = config.SCANS_DIR / res["scan_id"]
    saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert saved["scan_id"] == res["scan_id"] and (folder / "weight.json").exists()
    events.act("finish")
    events.screen("home")
    wait_for(lambda: client.kiosk.flow is None)
