"""Glue that must match the other domains' real shapes (checked against their code on 2026-09-27)."""
from __future__ import annotations

import io
import time
import wave

import numpy as np

from yq.server.adapters import Languages, to_wav
from yq.server.flows.scan_watch import ScanWatcher, normalize_detail
from yq.server.flows.story import sign_view
from yq.server import mock_assets


def test_sign_view_accepts_the_real_engine_state():
    real = {"hands_visible": True, "fps": 24.0, "mode": "letters", "sign_lang": "prl", "text": "hola con",
            "error": None, "buffer": "con", "letter": {"current": "n", "conf": 0.9, "progress": 0.4, "hand": "right"},
            "candidates": [{"text": "condor", "prob": 0.61, "kind": "word"}, {"text": "cono", "prob": 0.2, "kind": "word"}]}
    v = sign_view(real, None)
    assert v["candidates"][0] == {"text": "condor", "prob": 0.61, "kind": "word"}
    assert v["text"] == "hola con" and v["buffer"] == "con" and v["letter"]["current"] == "n"
    legacy = {"hands_visible": False, "candidates": [["a", 0.8], ["o", 0.1]], "buffer": "x"}

    class Eng:
        def text(self):
            return "casa"
    v = sign_view(legacy, Eng())
    assert v["candidates"][1] == {"text": "o", "prob": 0.1, "kind": ""} and v["text"] == "casa"


def test_to_wav_from_tts_tuple():
    wav = to_wav((np.zeros(1600, dtype=np.float32), 16000, "piper:x"))
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnframes(), w.getsampwidth()) == (16000, 1600, 2)
    assert to_wav(b"RIFF1234") == b"RIFF1234" and to_wav(None) is None


def test_language_rank_order(monkeypatch, fast):
    lang = Languages()
    rows = [{"code": "arl_Latn", "name_es": "Arabela", "popular": 0, "peru": True},
            {"code": "eng_Latn", "name_es": "Inglés", "popular": 9, "peru": False},
            {"code": "quy_Latn", "name_es": "Quechua", "popular": 2, "peru": True},
            {"code": "spa_Latn", "name_es": "Español", "popular": 1, "peru": True}]
    monkeypatch.setattr(lang._impl, "ui_list", lambda feature: rows)
    assert [x["code"] for x in lang.ui_list()] == ["spa_Latn", "quy_Latn", "arl_Latn", "eng_Latn"]


def test_normalize_box_details():
    assert normalize_detail("rti", {"led": 3})["light"] == "led3"
    d = normalize_detail("photogrammetry", {"camera": "B", "index": 6, "total": 24})
    assert d["platter_deg"] == 90.0 and d["light"] == "cob"


def test_watcher_reports_photos_and_thermal(tmp_path):
    folder = tmp_path / "scan-1"
    (folder / "photogrammetry").mkdir(parents=True)
    (folder / "thermal").mkdir()
    for name in ("camA_000.jpg", "camB_350.jpg", "background_camA.jpg"):
        (folder / "photogrammetry" / name).write_bytes(mock_assets.object_photo(0, "white", 64, 48))
    np.save(folder / "thermal" / "sequence.npy", np.stack([mock_assets.thermal_field(t) for t in (0.2, 1.0, 1.5)]))
    past = time.time() - 5
    import os
    for p in list((folder / "photogrammetry").iterdir()) + [folder / "thermal" / "sequence.npy"]:
        os.utime(p, (past, past))
    got = []
    w = ScanWatcher(folder, lambda rel: "/files/scans/scan-1/" + rel, lambda t, **kw: got.append(dict(kw, type=t)),
                    tmp_path / "cache", lambda n: "/files/uicache/scan-1/" + n)
    w.check()
    photos = [g for g in got if g["type"] == "scan_photo"]
    assert [(p["camera"], p["angle"]) for p in photos] == [("A", 0.0), ("B", 350.0)]
    thermal = [g for g in got if g["type"] == "scan_thermal"]
    assert thermal and (tmp_path / "cache" / "thermal_preview.png").exists()
    w.check()
    assert len(got) == 3                                       # nothing reported twice
