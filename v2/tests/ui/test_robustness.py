"""The kiosk never gets stuck: idle reset, cancel during slow work, kind errors with retry."""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from yq.common import config
from yq.common.contracts import DrawingResult, Transcript
from yq.server import mocks_art_box, settings
from tests.ui.conftest import wait_for


def test_idle_reset_warns_then_goes_home(client, events, monkeypatch):
    monkeypatch.setattr(config, "IDLE_RESET_SECONDS", 1.2)
    monkeypatch.setattr(settings, "IDLE_WARNING_SECONDS", 0.8)
    client.post("/api/start", json={"flow": "story"})
    events.screen("method")
    w = events.until(lambda m: m["type"] == "idle_warning", what="idle warning")
    assert 0 <= w["seconds"] <= 1
    home = events.screen("home", timeout=5)
    assert home["data"]["reset"] is True
    assert [m for m in events.all if m["type"] == "flow_end"][-1]["outcome"] == "idle"


def test_activity_keeps_the_visitor_screen(client, events, monkeypatch):
    monkeypatch.setattr(config, "IDLE_RESET_SECONDS", 1.0)
    monkeypatch.setattr(settings, "IDLE_WARNING_SECONDS", 0.5)
    client.post("/api/start", json={"flow": "story"})
    events.screen("method")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        events.ws.send_text('{"type": "activity"}')
        time.sleep(0.2)
    assert client.kiosk.flow is not None and client.kiosk.flow.screen == "method"
    client.post("/api/cancel")
    events.screen("home")


def test_cancel_while_a_subsystem_hangs(client, events, monkeypatch):
    hung = threading.Event()

    def slow_drawing(story, out_dir, on_progress=None):
        hung.set()
        time.sleep(30)                                    # a frozen Mac / generator
    monkeypatch.setattr(client.kiosk.sub.art, "make_drawing", slow_drawing)
    client.post("/api/start", json={"flow": "story", "method": "text"})
    events.screen("typing")
    events.act("submit_text", text="Un cóndor vuela.", lang="spa_Latn")
    events.screen("confirm")
    events.act("confirm")
    events.screen("consent")
    events.act("consent", publish=False)
    events.screen("making")
    assert hung.wait(5)
    t0 = time.time()
    assert client.post("/api/cancel").json()["ok"] is True
    events.screen("home", timeout=3)
    assert time.time() - t0 < 2.5
    wait_for(lambda: client.kiosk.flow is None)
    assert client.post("/api/start", json={"flow": "scan"}).status_code == 200   # usable again at once


def test_drawing_error_then_retry(client, events, monkeypatch):
    monkeypatch.setattr(mocks_art_box.MockArt, "fail", True)
    client.post("/api/start", json={"flow": "story", "method": "text"})
    events.screen("typing")
    events.act("submit_text", text="Una llama en la puna.")
    events.screen("confirm")
    events.act("confirm")
    events.screen("consent")
    events.act("consent", publish=True)
    err = events.screen("error", timeout=10)
    assert err["data"]["failed"] == "making" and err["data"]["message_es"]
    events.act("retry")
    events.screen("making")
    events.screen("done", timeout=20)


def test_empty_transcript_is_a_kind_error_back_to_the_mic(client, events, monkeypatch):
    monkeypatch.setattr(client.kiosk.sub.voice._mock, "transcribe",
                        lambda audio, lang="auto": Transcript(text="  ", lang="spa_Latn", engine="mock"))
    client.post("/api/start", json={"flow": "story", "method": "voice"})
    events.screen("voice_lang")
    events.act("choose_lang", code="spa_Latn")
    events.screen("voice_ready")
    events.act("record")
    err = events.screen("error", timeout=10)
    assert err["data"]["key"] == "err_no_speech"
    events.act("retry")
    events.screen("voice_ready")
    events.act("home")
    events.screen("home")


def test_stop_listening_button_ends_recording_early(client, events, monkeypatch):
    monkeypatch.setattr(settings, "MOCK_SPEED", 1.0)          # 5 s of mock speech unless stopped
    client.post("/api/start", json={"flow": "story", "method": "voice"})
    events.screen("voice_lang")
    events.act("choose_lang", code="spa_Latn")
    events.screen("voice_ready")
    events.act("record")
    events.screen("listening")
    time.sleep(0.7)
    t0 = time.time()
    assert client.post("/api/act/stop_listening").json()["ok"] is True
    events.screen("transcribing", timeout=3)
    assert time.time() - t0 < 1.5
    client.post("/api/cancel")
    events.screen("home", timeout=5)


def test_kiosk_exit_needs_the_password(client, monkeypatch):
    called = []
    monkeypatch.setattr("yq.server.app.close_kiosk_browser", lambda: called.append(1) or True)
    assert client.post("/api/kiosk/exit", json={"password": "nope"}).status_code == 403
    assert not called
    r = client.post("/api/kiosk/exit", json={"password": config.KIOSK_EXIT_PASSWORD})
    assert r.json() == {"ok": True, "closed": True} and called == [1]
    assert client.post("/api/admin/check", json={"password": "x"}).status_code == 403
    assert client.post("/api/admin/check", json={"password": config.KIOSK_EXIT_PASSWORD}).json()["ok"]


def test_files_cannot_escape_the_data_folders(client):
    (config.STORIES_DIR / "s1").mkdir(parents=True)
    (config.STORIES_DIR / "s1" / "a.txt").write_text("hi")
    assert client.get("/files/stories/s1/a.txt").text == "hi"
    for bad in ("/files/stories/../../etc/passwd", "/files/stories/%2e%2e/%2e%2e/etc/passwd",
                "/files/other/x", "/files/stories/s1/missing.png"):
        assert client.get(bad).status_code == 404, bad


def test_status_lists_every_subsystem(client):
    st = client.get("/status").json()
    subs = st["subsystems"]
    for name in ("voice", "languages", "sign", "art", "box", "camera", "printer", "hologram"):
        assert subs[name] == "mock:ui", (name, subs[name])
    assert set(st["reachable"]) == {"mac", "printer", "hologram"}
    assert st["publish"]["mode"] == "mock"


def test_languages_peruvian_first(client):
    langs = client.get("/api/languages?feature=asr").json()["languages"]
    peru = [x["peru"] for x in langs]
    assert peru == sorted(peru, reverse=True) and sum(peru) >= 5
    assert langs[0]["code"] == "spa_Latn" and {"quy_Latn", "ayr_Latn"} <= {x["code"] for x in langs if x["peru"]}


def test_real_modules_plug_in_without_changes(monkeypatch, fast):
    """Fake modules with the CONTRACTS.md §6 signatures are picked up instead of the UI mocks."""
    monkeypatch.setattr(settings, "UI_MOCKS", [""])
    calls = []
    asr = types.ModuleType("yq.voice.asr")
    asr.transcribe = lambda audio, lang="auto": calls.append(("asr", lang)) or Transcript("hola", "spa_Latn", "fake")
    drawing = types.ModuleType("yq.art.drawing")

    def make_drawing(story, out_dir, on_progress=None):
        calls.append(("art", story.story_id))
        return DrawingResult(story_id=story.story_id, image="", front_svg="f.svg", back_svg="b.svg")
    drawing.make_drawing = make_drawing
    scan = types.ModuleType("yq.box.scan")
    scan.preflight = lambda: {"ok": True, "problems_es": [], "weight_g": 1.0}
    scan.run_scan = lambda req, on_progress=None, cancel_event=None: calls.append(("scan", req.profile))
    for name, mod in (("yq.voice.asr", asr), ("yq.art.drawing", drawing), ("yq.box.scan", scan)):
        monkeypatch.setitem(sys.modules, name, mod)
    from yq.common.contracts import ScanRequest, StoryInput
    from yq.server import adapters
    v = adapters.Voice()
    assert v.parts["asr"][0] is asr and v.transcribe(None, "quy_Latn")["text"] == "hola"
    a = adapters.Art()
    assert a.mode == "yq.art" and a.make_drawing(StoryInput("s1", "text", "x", "spa_Latn"), "/tmp")["front_svg"]
    b = adapters.Box()
    assert b.mode.startswith("yq.box") and b.preflight()["ok"]
    b.run_scan(ScanRequest("x", "quick"))
    assert calls == [("asr", "quy_Latn"), ("art", "s1"), ("scan", "quick")]


@pytest.mark.parametrize("looks,vote,expected", [
    ([], None, False), ([{"covered": False}] * 3, None, True), ([{"covered": False}, {"covered": True}], None, True),
    ([{"covered": True}, {"covered": True}], None, False), ([{"covered": True}] * 2, True, True),
    ([{"covered": False}], False, False)])
def test_consent_decision(looks, vote, expected):
    from yq.server.camera import decide
    assert decide(looks, vote)[0] is expected


def test_look_at_detects_covered_mock_frames():
    from yq.server import mock_assets
    from yq.server.camera import look_at
    assert look_at(mock_assets.portrait_frame(0, covered=True))["covered"] is True
    assert look_at(mock_assets.portrait_frame(0, covered=False))["covered"] is False
