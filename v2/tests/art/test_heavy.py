"""Real models on the Mac (minutes, GBs of RAM). Run inside the memory lock:

    .venvs/art/bin/python -m yq.common.heavylock .venvs/art/bin/python -m pytest -m heavy tests/art/test_heavy.py
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module", autouse=True)
def _unload_after():
    yield
    from yq.macworker.modelmgr import models
    models.unload_all()


def test_llm_json_plan_spanish_and_quechua():
    from yq.macworker.models import art_plan
    p, es, en = art_plan.plan("Un cóndor volaba sobre los Apus nevados mientras una niña cuidaba sus llamas.",
                              "spa_Latn")
    joined = " ".join(p.elements).lower()
    assert "condor" in joined and "llama" in joined
    assert p.prompt and "condor" in p.prompt.lower() and en
    p2, es2, _ = art_plan.plan("Ñawpa pachapi huk kunturmi karqan, urqukunapa hawanta phawarqan.", "quy_Latn",
                               text_es="Hace mucho tiempo había un cóndor que volaba sobre las montañas.")
    assert "condor" in " ".join(p2.elements).lower()


def test_llm_plain_chat():
    from yq.macworker.models import llm
    out = llm.chat([{"role": "user", "content": "Reply with the single word: llama"}], max_tokens=10)
    assert "llama" in out.lower()


def test_vlm_reads_a_drawing():
    import cv2
    from PIL import Image
    from yq.macworker.models import vlm
    img = np.full((600, 600, 3), 255, np.uint8)
    cv2.circle(img, (200, 300), 120, (0, 0, 0), 6)
    pts = np.array([[380, 420], [560, 420], [470, 200]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 6)
    d = vlm.ask_json([Image.fromarray(img)], 'Which shapes are drawn? Reply {"shapes": ["..."]}',
                     schema={"shapes": list})
    got = " ".join(d["shapes"]).lower()
    assert "circle" in got and "triangle" in got


def test_generate_verify_and_vectorize():
    from yq.art import vectorize
    from yq.macworker.models import art_image, art_style, art_verify
    prompt = art_style.compose_prompt("An Andean condor with open wings flying over a snowy mountain")
    img = art_image.generate(prompt, 512, 704, seed=11)
    assert img.size == (512, 704)
    v = art_verify.verify(img, ["condor", "mountain"])
    assert v["verified"].get("condor") and 0 <= v["score"] <= 1
    tr = vectorize.trace(img)
    assert len(tr.strokes) > 20
