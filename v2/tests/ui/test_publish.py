"""Store-and-forward publishing into a TEMPORARY git repo (never the real repo or GitHub)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from yq.common import config
from yq.publish import gallery, settings as pset, sync

REAL_GALLERY = Path(__file__).resolve().parents[3] / "docs" / "index.html"
V1_ENTRY = {"id": "2026-06-11_14-22-10", "images": ["scene_1.png"], "photos": [], "portrait": None,
            "story": "STORY\n====\nHabía una vez...\n\nSCENES\n====\n1. x\n\nTITLE: Viejo\nLANG: spanish\n"}


def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """origin (bare) with main:/docs like GitHub Pages + the robot's separate clone."""
    seed, origin, clone = tmp_path / "seed", tmp_path / "origin.git", tmp_path / "archivo"
    (seed / "docs" / "stories" / V1_ENTRY["id"]).mkdir(parents=True)
    shutil.copy(REAL_GALLERY, seed / "docs" / "index.html")
    (seed / "docs" / "stories" / V1_ENTRY["id"] / "story.txt").write_text(V1_ENTRY["story"])
    (seed / "docs" / "stories" / V1_ENTRY["id"] / "scene_1.png").write_bytes(b"png")
    (seed / "docs" / "stories.json").write_text(json.dumps([V1_ENTRY]))
    git("init", "-q", "-b", "main", cwd=seed)
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "seed", cwd=seed)
    git("clone", "-q", "--bare", str(seed), str(origin), cwd=tmp_path)
    git("clone", "-q", str(origin), str(clone), cwd=tmp_path)
    monkeypatch.setattr(pset, "PUBLISH_REPO_DIR", clone)
    monkeypatch.setattr(pset, "PUBLISH_ENABLED", True)
    monkeypatch.setattr(config, "MOCK", False)
    monkeypatch.delenv("YQ_MOCK_PUBLISH", raising=False)
    from yq.server import settings as ui
    monkeypatch.setattr(ui, "UI_MOCKS", [""])
    return {"origin": origin, "clone": clone, "tmp": tmp_path}


def make_story(story_id, marker=".ready", lang="quy_Latn", portrait=True):
    d = Path(config.STORIES_DIR) / story_id
    d.mkdir(parents=True)
    meta = {"story": {"story_id": story_id, "source": "voice", "text": "Unay pachas…", "lang": lang,
                      "text_es": "Hace tiempo…", "sign_lang": ""}, "title": "El cóndor",
            "lang_name_es": "Quechua ayacuchano", "qr_url": gallery.story_url(story_id)}
    (d / "story.json").write_text(json.dumps(meta))
    (d / "story.txt").write_text(gallery.story_txt("Unay pachas…", "Hace tiempo…", "El cóndor", "", lang,
                                                   "Quechua ayacuchano", "voice"), encoding="utf-8")
    (d / "scene_1.png").write_bytes(b"\x89PNG fake")
    (d / "front.svg").write_text("<svg/>")
    (d / "narration_1.wav").write_bytes(b"RIFF")
    if portrait:
        (d / "storyteller_photo.jpg").write_bytes(b"\xff\xd8 fake")
    if marker:
        (d / marker).write_text("")
    return d


def remote_file(origin, path):
    return subprocess.run(["git", "show", "main:" + path], cwd=origin, capture_output=True, text=True).stdout


def test_publish_to_temp_repo_keeps_the_gallery_compatible(archive):
    make_story("story-20260926-201500-ab12")
    make_story("story-20260926-203000-cd34", marker=".private")
    make_story("story-20260926-204500-ef56", marker="")           # still being drawn
    assert sync.pending() == ["story-20260926-201500-ab12"]
    res = sync.sync()
    assert res["status"] == "published" and res["new"] == 1 and res["ids"] == ["2026-09-26_20-15-00_ab12"]
    entries = json.loads(remote_file(archive["origin"], "docs/stories.json"))
    assert [e["id"] for e in entries] == ["2026-09-26_20-15-00_ab12", V1_ENTRY["id"]]   # newest first
    new, old = entries
    assert old == V1_ENTRY                                         # v1 entries untouched
    for key in ("id", "images", "photos", "portrait", "story"):   # what docs/index.html reads
        assert key in new
    assert new["images"] == ["scene_1.png"] and new["portrait"] == "storyteller_photo.jpg"
    assert new["lang"] == "quy_Latn" and new["qr_url"].endswith("#2026-09-26_20-15-00_ab12") and new["v"] == 2
    web = "docs/stories/2026-09-26_20-15-00_ab12/"
    listing = subprocess.run(["git", "ls-tree", "--name-only", "main", web], cwd=archive["origin"],
                             capture_output=True, text=True).stdout.split()
    assert sorted(Path(p).name for p in listing) == ["info.json", "scene_1.png", "story.txt", "storyteller_photo.jpg"]
    assert (Path(config.STORIES_DIR) / "story-20260926-201500-ab12" / ".published").exists()
    assert sync.sync()["status"] == "nothing"
    assert "<title>Yachachiq" in remote_file(archive["origin"], "docs/index.html")     # page untouched


def test_gallery_page_parsing_of_v2_story_txt():
    """Re-implements the gallery page's JS parsing (soloHistoria, tituloDe, esQuechua) on v2 output."""
    txt = gallery.story_txt("Unay pachas huk kuntur…", "Hace mucho tiempo un cóndor…", "El cóndor", "Resumen",
                            "quy_Latn", "Quechua ayacuchano", "voice")
    i = txt.find("SCENES")
    story = re.sub(r"=+", "", re.sub(r"^STORY\s*", "", txt[:i])).strip()
    assert story.startswith("Unay pachas") and "Hace mucho tiempo" in story and "SCENES" not in story
    assert re.search(r"^TITLE:\s*(.+)$", txt, re.M).group(1) == "El cóndor"
    assert re.search(r"^LANG:\s*quechua", txt, re.M | re.I)
    page = REAL_GALLERY.read_text(encoding="utf-8")
    for token in ('txt.indexOf("SCENES")', "TITLE:", "LANG:\\s*quechua", "h.portrait", "h.images", "h.photos"):
        assert token in page, "the gallery page changed: re-check compatibility (%s)" % token


def test_offline_keeps_stories_waiting(archive):
    make_story("story-20260926-201500-ab12")
    shutil.rmtree(archive["origin"])                               # no internet / GitHub unreachable
    res = sync.sync()
    assert res["status"] == "offline" and res["new"] == 1
    assert sync.pending() == ["story-20260926-201500-ab12"]


def test_modes_off_and_mock(archive, monkeypatch):
    make_story("story-20260926-201500-ab12")
    monkeypatch.setattr(pset, "PUBLISH_ENABLED", False)
    assert sync.sync()["status"] == "off"
    monkeypatch.setattr(pset, "PUBLISH_ENABLED", True)
    monkeypatch.setattr(config, "MOCK", True)
    assert sync.sync() == {"status": "mock", "new": 1, "t": sync.last_result["t"]}
    assert sync.pending()                                          # mock never marks anything published


def test_sync_later_runs_in_background(archive):
    make_story("story-20260926-201500-ab12")
    got = []
    res = sync.sync_later(on_done=got.append)
    assert res == {"status": "queued", "new": 1}
    from tests.ui.conftest import wait_for
    wait_for(lambda: got, timeout=30)
    assert got[0]["status"] == "published"


def test_web_id_and_url():
    assert gallery.web_id("story-20260926-201500-ab12") == "2026-09-26_20-15-00_ab12"
    assert gallery.web_id("weird/../id") == "weird____id"
    assert gallery.story_url("story-20260926-201500-ab12", "https://e.org/y") == "https://e.org/y/#2026-09-26_20-15-00_ab12"
