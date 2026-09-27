"""Printer and hologram HTTP clients against their mock servers on a real local port."""
from __future__ import annotations

import time

import pytest

from yq.hologram.client import HologramClient, HologramError, build_story, package_files
from yq.hologram.mock import HologramSim
from yq.hologram.mock_server import create_app as hologram_app
from yq.printer.client import PrinterClient, PrinterError, normalize
from yq.printer.mock import PrinterSim
from yq.printer.mock_server import create_app as printer_app
from yq.server import mock_assets
from tests.ui.conftest import serve

FRONT = mock_assets.svg_paths(mock_assets.drawing_polylines())
BACK = mock_assets.back_svg("Título", "Había una vez…", "https://x/#1")


def test_printer_job_goes_through_all_stages():
    sim = PrinterSim(speed=0.01)
    with serve(printer_app(sim)) as url:
        pc = PrinterClient(url)
        assert pc.available()
        job = pc.submit("story-1", FRONT, BACK, front_gcode="G0 X0\n", qr_url="https://x/#1")
        assert job["id"] == "p0001"
        stored = sim.jobs["p0001"]
        assert stored["job"]["flip"] == "long-edge" and stored["job"]["paper"] == {"w_mm": 210.0, "h_mm": 297.0}
        assert set(stored["files"]) == {"front.svg", "back.svg", "front.gcode", "job.json"}
        seen = []
        final = pc.wait(job["id"], on_status=lambda s: seen.append(s["status"]), poll_s=0.02, timeout=10)
        assert final["status"] == "done" and final["progress"] == 1.0
        assert seen.index("drawing_front") < seen.index("done")


def test_printer_rejects_svg_without_millimetres():
    with serve(printer_app(PrinterSim(0.01))) as url:
        with pytest.raises(PrinterError, match="HTTP 400"):
            PrinterClient(url).submit("s", '<svg width="210" height="297"></svg>', BACK)


def test_printer_cancel_and_unknown_job():
    with serve(printer_app(PrinterSim(1.0))) as url:
        pc = PrinterClient(url)
        jid = pc.submit("s", FRONT, BACK)["id"]
        assert pc.cancel(jid) and pc.status(jid)["status"] == "cancelled"
        assert pc.status("nope")["status"] == "sent"          # printers without status: tolerated


def test_printer_unreachable_is_fast_and_clear():
    pc = PrinterClient("http://127.0.0.1:9", timeout=1.0)
    t0 = time.time()
    assert pc.available() is False
    with pytest.raises(PrinterError, match="impresora no disponible"):
        pc.submit("s", FRONT, BACK)
    assert pc.status("x")["status"] == "unreachable"
    assert time.time() - t0 < 5


def test_printer_status_normalization():
    assert normalize({"state": "Printing_Front", "progress": 40})["status"] == "drawing_front"
    assert normalize({"state": "printing_front", "progress": 40})["progress"] == 0.4
    assert normalize({"status": "finished"})["status"] == "done"
    assert normalize({"status": "weird_state"})["status"] == "weird_state"


def test_hologram_package_roundtrip(tmp_path):
    img = tmp_path / "scene_1.png"
    img.write_bytes(mock_assets.drawing_png(mock_assets.drawing_polylines()))
    wav = tmp_path / "narration_1.wav"
    wav.write_bytes(mock_assets.wav_tone(0.5))
    scenes = [{"image": str(img), "caption_es": "Un cóndor", "audio": str(wav)}]
    story = build_story("story-1", "El cóndor", "quy_Latn", "Unay pachas…", "Hace tiempo…", scenes)
    assert story["scenes"] == [{"image": "scene_1.png", "caption_es": "Un cóndor", "audio": "narration_1.wav"}]
    sim = HologramSim(speed=0.01)
    with serve(hologram_app(sim)) as url:
        hc = HologramClient(url)
        assert hc.available()
        ans = hc.send(story, package_files(scenes))
        assert ans["ok"] and ans["id"] == "story-1"
        assert sim.stories["story-1"]["files"] == {"scene_1.png": img.stat().st_size, "narration_1.wav": wav.stat().st_size}
        assert hc.status()["playing"] in ("story-1", None)
        with pytest.raises(HologramError, match="faltan archivos"):
            hc.send(story, {"scene_1.png": b"x"})


def test_hologram_server_validates_story_json():
    import requests
    with serve(hologram_app(HologramSim(0.01))) as url:
        r = requests.post(url + "/stories", files=[("story.json", ("story.json", b'{"story_id":"x"}', "application/json"))])
        assert r.status_code == 400 and "needs" in r.text
        assert HologramClient("http://127.0.0.1:9", timeout=1).available() is False
