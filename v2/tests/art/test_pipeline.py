"""Mac routes with mocked LLM/VLM/generator, JSON helpers, and make_drawing end to end."""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from yq.common import config
from yq.common.contracts import DrawingResult, StoryInput


@pytest.fixture
def mocked(monkeypatch):
    for name in ("LLM", "VLM", "IMAGEGEN"):
        monkeypatch.setenv("YQ_MOCK_" + name, "1")
    from yq.macworker.jobs import jobs
    jobs.root = config.JOBS_DIR
    from yq.macworker.app import create_app
    return TestClient(create_app())


def _wait(c, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get("/jobs/" + jid).json()
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise AssertionError("job timed out")


def test_health_lists_art(mocked):
    h = mocked.get("/health").json()
    assert h["domains"].get("routes_art") == "ok", h["domains"]
    assert "image" in h["job_kinds"]


def test_story_clean_and_plan(mocked):
    r = mocked.post("/story/clean", json={"text": "  un condor   volaba ", "lang": "spa_Latn"}).json()
    assert r["text"] == "Un condor volaba"
    r = mocked.post("/story/plan", json={"text": "Un cóndor volaba sobre las montañas y una llama miraba.",
                                         "lang": "spa_Latn"}).json()
    plan = r["plan"]
    assert plan["elements"] and "condor" in " ".join(plan["elements"]).lower()
    assert plan["prompt"] and r["text_es"].startswith("Un cóndor")
    assert mocked.post("/story/plan", json={"text": ""}).status_code == 400


def test_image_job_with_mocks(mocked):
    plan = mocked.post("/story/plan", json={"text": "A condor above the mountains and the sun",
                                            "lang": "eng_Latn"}).json()["plan"]
    jid = mocked.post("/jobs", data={"kind": "image", "params": json.dumps(
        {"plan": plan, "width": 384, "height": 544, "seed": 3, "max_attempts": 2, "verify": True})}).json()["id"]
    st = _wait(mocked, jid)
    assert st["status"] == "done", st
    res = st["result"]
    assert res["image"] == "image.png" and res["attempts"] >= 1 and 0 <= res["score"] <= 1
    assert set(res["verified"]) == set(plan["elements"])
    assert "image.png" in st["files"] and "verify.json" in st["files"]
    png = mocked.get("/jobs/%s/files/image.png" % jid)
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"


def test_json_extraction_is_robust():
    from yq.macworker.models.art_runtime import extract_json, strip_think
    assert extract_json('Sure! ```json\n{"a": 1, "b": [1,2,],}\n```') == {"a": 1, "b": [1, 2]}
    assert extract_json('<think>hmm {"x": 0}</think>{"ok": True, "n": None}') == {"ok": True, "n": None}
    assert extract_json('prefix {"a": {"b": "c}"}} suffix') == {"a": {"b": "c}"}}
    assert extract_json('{"a": [1, {"b": 2}') == {"a": [1, {"b": 2}]}
    assert strip_think("<think>x</think> hola") == "hola"
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_llm_json_mode_retries(monkeypatch):
    from yq.macworker.models import llm
    replies = iter(["I think the answer is yes.", '{"title": 5}', '{"title": "El cóndor"}'])
    seen = []
    monkeypatch.setattr(llm, "_raw", lambda msgs, mt, temp, key: (seen.append(msgs), next(replies))[1])
    out = llm.chat([{"role": "user", "content": "plan"}], json_mode=True, schema={"title": str})
    assert json.loads(out) == {"title": "El cóndor"}
    assert len(seen) == 3 and "not valid" in seen[-1][-1]["content"]
    assert "JSON" in seen[0][0]["content"]


def test_llm_and_vlm_mocks(monkeypatch):
    monkeypatch.setenv("YQ_MOCK_LLM", "1")
    monkeypatch.setenv("YQ_MOCK_VLM", "1")
    from PIL import Image
    from yq.macworker.models import llm, vlm
    assert json.loads(llm.chat([{"role": "user", "content": "hola"}], json_mode=True))["mock"] is True
    assert vlm.ask([Image.new("RGB", (64, 64))], "describe").startswith("[mock vlm] 1 image")


def test_verify_metrics_flag_bad_pictures():
    from PIL import Image
    from yq.macworker.models import art_verify
    white = Image.new("RGB", (512, 700), "white")
    assert "too_empty" in art_verify.problems_from(art_verify.metrics(white))
    black = Image.new("RGB", (512, 700), "black")
    assert "too_dark" in art_verify.problems_from(art_verify.metrics(black))
    orange = Image.new("RGB", (512, 700), (240, 120, 30))
    assert "not_line_art" in art_verify.problems_from(art_verify.metrics(orange))


def _check_outputs(r: DrawingResult):
    for p in (r.image, r.front_svg, r.back_svg, r.front_gcode, r.back_gcode):
        assert Path(p).is_file() and Path(p).stat().st_size > 0, p
    root = ET.fromstring(Path(r.front_svg).read_text().split("?>", 1)[1])
    assert root.attrib["width"].endswith("mm")
    out = Path(r.front_svg).parent
    assert (out / "front.png").is_file() and (out / "back.png").is_file()
    assert r.strokes > 5 and r.pen_mm > 100 and r.est_minutes > 0
    assert r.qr_url.endswith("story-test-1/")


def test_make_drawing_offline_fallback(tmp_path, monkeypatch):
    from yq.art.drawing import make_drawing
    monkeypatch.setenv("YQ_MOCK_MAC", "1")
    seen = []
    s = StoryInput(story_id="story-test-1", source="voice", lang="quy_Latn",
                   text="Ñawpa pachapi huk kuntursi urqu hawanta phawarqan, llamakunata qhawaspa.",
                   text_es="Hace mucho tiempo un cóndor volaba sobre la montaña mirando a las llamas.")
    r = make_drawing(s, tmp_path / "d", on_progress=seen.append)
    _check_outputs(r)
    rep = json.loads((tmp_path / "d" / "drawing.json").read_text())
    assert rep["source"] == "offline" and "condor" in rep["front"]["motifs"]
    assert seen[-1].fraction == 1.0 and all(p.message_es for p in seen)


def test_make_drawing_through_mocked_mac(tmp_path, monkeypatch, mocked):
    """Full Mac path: plan + image job + download, over the real HTTP app (models mocked)."""
    from yq.common import macclient
    from yq.art import drawing

    class Resp:                                     # httpx response with the requests API used by MacClient
        def __init__(self, r):
            self.r, self.status_code, self.content, self.text = r, r.status_code, r.content, r.text
            self.ok = r.is_success

        def json(self):
            return self.r.json()

        def raise_for_status(self):
            self.r.raise_for_status()

    class LocalClient(macclient.MacClient):
        def _request(self, method, path, timeout=None, **kw):
            return Resp(mocked.request(method, path, **kw))

    monkeypatch.setattr(macclient, "_client", LocalClient(urls=["http://test"]))
    s = StoryInput(story_id="story-test-1", source="text", lang="spa_Latn",
                   text="Una niña y su llama caminaban hacia el río bajo el sol.")
    r = drawing.make_drawing(s, tmp_path / "m", use_mac=True)
    _check_outputs(r)
    rep = json.loads((tmp_path / "m" / "drawing.json").read_text())
    assert rep["source"].startswith("mac:") and r.verified and r.attempts >= 1
    assert rep["front"]["travel_mm"] <= rep["front"]["travel_naive_mm"]


def test_prompt_style_roundtrip_and_emphasis():
    from yq.macworker.models import art_style
    scene = "An Andean condor flying over snowy mountains"
    for b in ("z-image-turbo", "flux2-klein-4b", "comfyui"):
        p = art_style.compose_prompt(scene, b)
        assert p.startswith(("An Andean", "a simple line drawing of An Andean"))
        assert art_style.scene_of(p, b) == scene
    assert art_style.negative("comfyui") and not art_style.negative("z-image-turbo")
    e = art_style.emphasize(scene, ["llama"], ["text", "frame"], 1)
    assert e.startswith("Clearly showing llama") and "no letters" in e and "no border" in e


def test_image_job_falls_back_to_next_backend(monkeypatch, tmp_path):
    from yq.macworker import routes_art
    from yq.macworker.jobs import JobManager
    from yq.macworker.models import art_image
    monkeypatch.setenv("YQ_MOCK_VLM", "1")
    calls = []

    def fake_generate(prompt, w, h, seed=None, backend=None, negative="", steps=None, **kw):
        calls.append(backend)
        if backend == "z-image-turbo":
            raise RuntimeError("model not downloaded")
        return art_image.mock_image(prompt, w, h, seed or 1)

    monkeypatch.setattr(art_image, "generate", fake_generate)
    jm = JobManager(root=tmp_path)
    jm.register("image", routes_art.image_job)
    plan = {"title": "t", "title_es": "t", "summary_es": "", "subject": "condor", "elements": ["condor"],
            "prompt": "A condor over mountains. style", "negative": ""}
    job = jm.run_inline("image", {"plan": plan, "width": 256, "height": 352, "backend": "z-image-turbo",
                                  "max_attempts": 1})
    assert job.status == "done", job.error
    assert calls[:2] == ["z-image-turbo", "comfyui"] and job.result["backend"] == "comfyui"


def test_unpublished_story_qr_points_to_gallery(tmp_path, monkeypatch):
    from yq.art import qr
    from yq.art.drawing import make_drawing
    monkeypatch.setenv("YQ_MOCK_MAC", "1")
    assert qr.story_url("story-9") == config.PUBLIC_BASE_URL + "story-9/"
    assert qr.story_url("story-9", published=False) == config.PUBLIC_BASE_URL
    s = StoryInput(story_id="story-private-1", source="voice", lang="spa_Latn", text="Un zorro miraba la luna.")
    r = make_drawing(s, tmp_path / "p", published=False)
    assert r.qr_url == config.PUBLIC_BASE_URL
    rep = json.loads((tmp_path / "p" / "drawing.json").read_text())
    assert rep["published"] is False and rep["title"]


def test_plan_salvages_almost_json(monkeypatch):
    from yq.macworker.models import art_plan, llm
    bad = ('{"title": "El Cóndor", "title_es": "El Cóndor", "summary_es": "El abuelo dice "hola"",\n'
           ' "subject": "condor", "elements": ["condor", "snowy mountain"],\n'
           ' "scene": "A condor with a white ruff over a snowy mountain"\n "mood": "calm"')

    def fail(*a, **k):
        e = ValueError("bad json")
        e.last_text = bad
        raise e
    monkeypatch.setattr(llm, "chat_json", fail)
    p, es, en = art_plan.plan("El cóndor vuela sobre el Apu.", "spa_Latn")
    assert p.elements == ["condor", "snowy mountain"] and p.prompt.startswith("A condor with a white ruff")
    assert es == "El cóndor vuela sobre el Apu."


def test_image_job_escalates_to_the_accurate_generator(monkeypatch, tmp_path):
    """Attempt 1 with the fast default; when the check fails, retries use ESCALATE_BACKEND."""
    from yq.macworker import routes_art
    from yq.macworker.jobs import JobManager
    from yq.macworker.models import art_image, art_verify
    calls, checks = [], iter([False, True])

    def fake_generate(prompt, w, h, seed=None, backend=None, negative="", steps=None, **kw):
        calls.append(backend)
        return art_image.mock_image(prompt, w, h, seed or 1)

    def fake_verify(img, elements, use_vlm=True, culture=""):
        ok = next(checks)
        v = {"verified": {e: ok for e in elements}, "line_art": 9, "text": False, "frame": False, "note": "",
             "metrics": {}, "problems": [], "missing": [] if ok else list(elements)}
        v["score"] = 1.0 if ok else 0.4
        return v

    monkeypatch.setattr(art_image, "generate", fake_generate)
    monkeypatch.setattr(art_verify, "verify", fake_verify)
    monkeypatch.setattr(routes_art, "ESCALATE_BACKEND", "z-image-turbo")
    jm = JobManager(root=tmp_path)
    jm.register("image", routes_art.image_job)
    plan = {"title": "t", "title_es": "t", "summary_es": "", "subject": "condor", "elements": ["condor"],
            "cultural_notes": ["culture: andean"], "prompt": "A condor over mountains. style", "negative": ""}
    job = jm.run_inline("image", {"plan": plan, "width": 256, "height": 352, "max_attempts": 3})
    assert job.status == "done", job.error
    assert calls == [art_image.BACKEND, "z-image-turbo"]          # stopped as soon as the check passed
    assert job.result["backend"] == "z-image-turbo" and job.result["attempts"] == 2
