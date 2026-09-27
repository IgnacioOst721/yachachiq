"""Story flow end-to-end through the real server (TestClient + WebSocket), UI mocks."""
from __future__ import annotations

import json
from pathlib import Path

from yq.common import config
from tests.ui.conftest import wait_for


def _start(client, **body):
    r = client.post("/api/start", json=dict({"flow": "story"}, **body))
    assert r.status_code == 200, r.text


def _story_dir() -> Path:
    dirs = [d for d in Path(config.STORIES_DIR).iterdir() if d.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_voice_story_auto_language_relang_and_publish(client, events):
    assert events.hello["screen"] == "home"
    _start(client)
    events.screen("method")
    events.act("choose", method="voice")
    m = events.screen("voice_lang")
    assert m["data"]["languages_url"].startswith("/api/languages")
    events.act("choose_lang", code="auto")
    events.screen("voice_ready")
    events.act("record")
    events.screen("listening")
    events.until(lambda m: m["type"] == "level", what="mic level")
    c = events.screen("confirm")
    assert c["data"]["source"] == "voice" and c["data"]["lang"] == "spa_Latn"
    assert [x[0] for x in c["data"]["candidates"]][:2] == ["spa_Latn", "quy_Latn"]
    # the visitor says: it was Quechua -> re-transcribe with that language, Spanish shown under it
    events.act("relang", code="quy_Latn")
    events.screen("transcribing")
    c = events.screen("confirm")
    assert c["data"]["lang"] == "quy_Latn" and c["data"]["translating"] is True
    upd = events.until(lambda m: m["type"] == "screen_update" and "text_es" in m["data"], what="translation")
    assert "cóndor" in upd["data"]["text_es"]
    events.act("confirm")
    events.screen("consent")
    events.act("consent", publish=True)
    res = events.screen("consent_result")
    assert res["data"]["publish"] is True and "botón" in res["data"]["reason"]
    events.screen("making")
    show = events.screen("showtime", timeout=20)
    assert show["data"]["image"].endswith("/scene_1.png")
    events.until(lambda m: m["type"] == "printer", what="printer status")
    done = events.screen("done", timeout=20)
    assert done["data"]["public"] is True and "#20" in done["data"]["qr_url"]
    d = _story_dir()
    for f in ("story.json", "story.txt", "scene_1.png", "storyteller_photo.jpg", ".ready", "narration_1.wav"):
        assert (d / f).exists(), f
    txt = (d / "story.txt").read_text(encoding="utf-8")
    assert "LANG: quechua ayacuchano (quy_Latn)" in txt and "— En español —" in txt
    meta = json.loads((d / "story.json").read_text(encoding="utf-8"))
    assert meta["story"]["lang"] == "quy_Latn" and meta["consent"]["publish"] is True
    assert client.get(done["data"]["image"]).status_code == 200
    said = client.kiosk.sub.voice._mock.said
    assert said and all(lang == "quy_Latn" for _, lang in said)       # mock language list says quy has TTS
    events.act("finish")
    home = events.screen("home")
    assert home["data"]["reset"] is False


def test_sign_story_with_candidates_and_private(client, events):
    _start(client, method="sign")
    m = events.screen("sign_lang")
    assert {s["code"] for s in m["data"]["sign_langs"]} == {"ase", "prl", "ils"}
    events.act("choose_sign", code="prl")
    s = events.screen("signing")
    assert s["data"]["preview"] == "/sign/preview.mjpeg" and s["data"]["can_space"] is True
    ev = events.until(lambda m: m["type"] == "sign" and m.get("candidates"), what="sign candidates")
    c = ev["candidates"][0]
    assert isinstance(c["text"], str) and 0 < c["prob"] <= 1 and ev["buffer"] and ev["letter"]["current"]
    events.act("sign_accept", index=1)                    # the visitor taps the second chip
    events.until(lambda m: m["type"] == "sign" and m.get("text"), what="text")
    r = client.get("/sign/preview.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    r = client.get("/sign/preview.mjpeg?max_frames=2")
    assert r.content.count(b"--frame") == 2
    events.act("sign_backspace")
    events.act("sign_accept", index=0)
    wait_for(lambda: client.kiosk.flow.engine and client.kiosk.flow.engine.text())
    events.act("sign_done")
    c = events.screen("confirm")
    assert c["data"]["source"] == "sign" and c["data"]["sign_lang"] == "prl" and c["data"]["text"]
    assert c["data"]["lang"] == "spa_Latn" and c["data"]["translating"] is False
    events.act("confirm")
    events.screen("consent")
    events.act("consent", publish=False)
    assert events.screen("consent_result")["data"]["publish"] is False
    done = events.screen("done", timeout=20)
    assert done["data"]["public"] is False and done["data"]["qr_url"] == config.PUBLIC_BASE_URL
    d = _story_dir()
    assert (d / ".private").exists() and not (d / ".ready").exists()
    assert not (d / "storyteller_photo.jpg").exists()
    assert "SIGN: Lengua de Señas Peruana (LSP) (prl)" in (d / "story.txt").read_text(encoding="utf-8")


def test_text_story_edit_and_covered_camera(client, events):
    client.kiosk.sub.camera.covered = True               # the visitor covers the lens: no publishing
    _start(client, method="text")
    events.screen("typing")
    events.act("submit_text", text="Once upon a time a condor lived high in the mountains. Every morning it "
                                   "flew over the lake and greeted the fishermen. One day it found a lost llama "
                                   "and helped it return to its family.", lang="eng_Latn")
    c = events.screen("confirm")
    assert c["data"]["translating"] is True and c["data"]["lang_name"] == "Inglés"
    events.until(lambda m: m["type"] == "screen_update" and m["data"].get("text_es"), what="text_es")
    events.act("edit_text", text="A condor and a llama became friends.")
    upd = events.until(lambda m: m["type"] == "screen_update" and m["data"].get("text") ==
                       "A condor and a llama became friends.", what="edited text")
    assert upd["data"]["translating"] is True
    events.act("confirm")
    events.screen("consent")
    tick = events.until(lambda m: m["type"] == "consent_tick", what="tick")
    assert tick["camera"] is True
    res = events.screen("consent_result", timeout=10)
    assert res["data"] == {"publish": False, "reason": "cámara tapada"}
    events.screen("done", timeout=20)
    meta = json.loads((_story_dir() / "story.json").read_text(encoding="utf-8"))
    assert meta["story"]["text"] == "A condor and a llama became friends." and meta["story"]["source"] == "text"


def test_back_navigation_and_home(client, events):
    _start(client)
    events.screen("method")
    events.act("choose", method="voice")
    events.screen("voice_lang")
    events.act("back")
    events.screen("method")
    r = client.post("/api/cancel")
    assert r.json()["ok"] is True
    events.screen("home")
    wait_for(lambda: client.kiosk.flow is None)
    assert client.post("/api/act/choose", json={"method": "voice"}).status_code == 409


def test_second_flow_is_refused_while_busy(client, events):
    _start(client)
    events.screen("method")
    r = client.post("/api/start", json={"flow": "scan"})
    assert r.status_code == 409 and r.json()["error"] == "ocupado"
