"""Optional "¿Dónde lo encontraron?" step (CONTRACTS.md §9) and the QR of declined stories."""
from __future__ import annotations

import json
import sys
import types

from yq.common import config
from yq.common.contracts import DrawingResult, StoryInput


def _to_context(client, events):
    assert client.post("/api/start", json={"flow": "scan"}).status_code == 200
    events.screen("intro")
    events.act("choose_profile", profile="quick")
    events.act("start")
    c = events.screen("context")
    assert len(c["data"]["regions"]) == 11 and c["data"]["max_chars"] == 200
    return c


def _results(events):
    events.screen("preflight")
    events.screen("scanning")
    return events.screen("results", timeout=40)["data"]["result"]


def test_typed_place_and_region_chip_reach_the_box(client, events):
    _to_context(client, events)
    events.act("set_context", region_hint="costa_norte")
    upd = events.until(lambda m: m["type"] == "screen_update" and m["data"].get("region_hint") == "costa_norte",
                       what="chip")
    assert upd["screen"] == "context"
    events.act("continue", found_where="En una huaca cerca de Trujillo " + "x" * 300, region_hint="costa_norte",
               lang="es")
    res = _results(events)
    ctx = res["context"]
    assert ctx["region_hint"] == "costa_norte" and ctx["lang"] == "spa_Latn" and ctx["notes"] == ""
    assert ctx["found_where"].startswith("En una huaca") and len(ctx["found_where"]) == 200
    meta = json.loads((config.SCANS_DIR / res["scan_id"] / "meta.json").read_text(encoding="utf-8"))
    assert meta["context"] == ctx                                      # ScanRequest.context -> meta.json
    ident = res["identification"]
    assert "costa norte" in ident["context_effect_es"] and ident["image_only"]["culture"] == ident["culture"]


def test_dictated_place_is_shown_for_confirmation(client, events):
    _to_context(client, events)
    events.act("dictate", ui_lang="qu")                                # Quechua UI -> auto-detect
    events.until(lambda m: m["type"] == "screen_update" and m["data"].get("recording") is True, what="recording")
    done = events.until(lambda m: m["type"] == "screen_update" and m["data"].get("dictated"), what="dictation")
    text = done["data"]["found_where"]
    assert text and len(text) <= 200
    assert client.kiosk.flow._ctx["lang"] == "spa_Latn"                # the transcript's language
    events.act("continue", found_where=text, region_hint="")          # no "lang": keep the transcript's
    res = _results(events)
    assert res["context"]["found_where"] == text and res["context"]["lang"] == "spa_Latn"


def test_skip_sends_no_context(client, events):
    _to_context(client, events)
    events.act("set_context", region_hint="selva")
    events.act("skip")
    res = _results(events)
    assert res["context"] == {}
    assert res["identification"]["context_effect_es"] == "" and res["identification"]["image_only"] is None


def test_place_that_changes_the_answer_keeps_the_image_only_answer(client, events):
    _to_context(client, events)
    events.act("continue", found_where="cerca de Chan Chan", region_hint="", lang="es")
    ident = _results(events)["identification"]
    assert ident["culture"] == "Chimú" and ident["image_only"]["culture"] == "Moche"   # UI shows both


def test_back_returns_to_intro(client, events):
    _to_context(client, events)
    events.act("back")
    events.screen("intro")


def test_private_story_qr_points_to_the_gallery(client, events):
    client.post("/api/start", json={"flow": "story", "method": "text"})
    events.screen("typing")
    events.act("submit_text", text="Un cóndor vuela sobre el Colca.")
    events.screen("confirm")
    events.act("confirm")
    events.screen("consent")
    events.act("consent", publish=False)
    done = events.screen("done", timeout=20)
    assert done["data"]["qr_url"] == config.PUBLIC_BASE_URL and done["data"]["public"] is False
    d = next(p for p in config.STORIES_DIR.iterdir() if p.is_dir())
    back = (d / "art" / "back.svg").read_text(encoding="utf-8")
    assert config.PUBLIC_BASE_URL in back and d.name not in back      # nothing points to this story


def test_adapter_makes_real_art_print_the_gallery_qr(monkeypatch, fast):
    """ART without a `published` parameter: its QR format is switched to the gallery for that call."""
    monkeypatch.setattr(fast, "UI_MOCKS", [""])
    art_settings = types.ModuleType("yq.art.settings")
    art_settings.QR_URL_FORMAT = "{base}{story_id}"
    drawing = types.ModuleType("yq.art.drawing")
    seen = []

    def make_drawing(story, out_dir, on_progress=None):
        url = art_settings.QR_URL_FORMAT.format(base=config.PUBLIC_BASE_URL, story_id=story.story_id)
        seen.append(url)
        return DrawingResult(story_id=story.story_id, image="", front_svg="f", back_svg="b", qr_url=url)
    drawing.make_drawing = make_drawing
    monkeypatch.setitem(sys.modules, "yq.art.settings", art_settings)
    monkeypatch.setitem(sys.modules, "yq.art.drawing", drawing)
    from yq.server.adapters import Art
    art = Art()
    s = StoryInput("story-1", "text", "x", "spa_Latn")
    assert art.make_drawing(s, "/tmp", published=False)["qr_url"] == config.PUBLIC_BASE_URL
    assert art_settings.QR_URL_FORMAT == "{base}{story_id}"            # restored
    assert art.make_drawing(s, "/tmp", published=True)["qr_url"].endswith("story-1")
    assert seen == [config.PUBLIC_BASE_URL, config.PUBLIC_BASE_URL + "story-1"]
