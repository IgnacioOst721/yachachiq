"""The kiosk only offers story languages the robot can hear and understand."""


def _codes(client, feature):
    return [x["code"] for x in client.get("/api/languages?feature=" + feature).json()["languages"]]


def test_story_and_translate_lists(client):
    story, typed, anyl = _codes(client, "story"), _codes(client, "translate"), _codes(client, "")
    assert story and typed and set(story) <= set(anyl) and set(typed) <= set(anyl)
    assert story[0] == "spa_Latn" and "quy_Latn" in story


def test_mock_story_list_needs_hearing_and_understanding():
    from yq.server.mocks import MockLanguages
    rows = MockLanguages().ui_list("story")
    assert rows and all(r["asr"] and r["translate"] for r in rows)
    assert MockLanguages().ui_list("any") == MockLanguages().ui_list("")


def test_story_flow_asks_for_understood_languages():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "yq/server/flows/story.py").read_text()
    assert 'languages_url="/api/languages?feature=story"' in src
    assert 'languages_url="/api/languages?feature=translate"' in src
