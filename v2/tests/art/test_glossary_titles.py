"""Andean visual glossary, abstract-element filter and title checks (fixes from the first real runs:
the condor came out as an eagle, and a Quechua story got its last word "Qhawaspa" as title)."""
from PIL import Image

from yq.macworker.models import art_glossary as G
from yq.macworker.models import art_plan, art_style


def test_condor_gets_its_real_look_and_a_strict_check():
    s = G.enrich_scene("A large condor soars above the Andes", ["condor"], "andean")
    assert "not an eagle" in s and "bald" in s and "white ring" in s
    hint = G.verify_hint("condor")
    assert "bald head" in hint and "eagle" in hint
    assert G.verify_hint("kuntur") == hint


def test_village_is_andean_only_for_andean_stories():
    andean = G.enrich_scene("a small village below the mountain", ["village"], "andean")
    other = G.enrich_scene("a small village next to the lighthouse", ["village"], "english")
    assert "adobe" in andean and "no pine trees" in andean
    assert "adobe" not in other


def test_snowy_apus():
    assert "snow caps" in G.enrich_scene("a condor above the snowy apus", [], "andean")
    assert "snow caps" in G.enrich_scene("x", ["snowy mountain"], "")


def test_abstract_elements_are_dropped():
    assert G.drawable_elements(["condor", "snowy mountain", "village", "morning sky"]) == \
        ["condor", "snowy mountain", "village"]
    assert G.drawable_elements(["condor", "llama", "mountain", "sky"]) == ["condor", "llama", "mountain"]
    assert G.drawable_elements(["moon", "fox"]) == ["moon", "fox"]
    assert G.drawable_elements(["sky"]) == ["sky"]            # never leave the list empty


def test_good_title_rejects_the_copied_last_word():
    story = "Ñawpa pachapi huk kunturmi karqan. Payqa sapa p'unchaw phawarqan, llamakunata qhawaspa."
    assert not art_plan.good_title("Qhawaspa", story)
    assert art_plan.good_title("Kuntur", story)
    assert art_plan.good_title("El Cóndor y los Apus")
    assert not art_plan.good_title("")
    assert not art_plan.good_title(" ".join(["palabra"] * 12))


def test_fix_titles_quechua(monkeypatch):
    story = "Ñawpa pachapi huk kunturmi karqan. Payqa sapa p'unchaw phawarqan, llamakunata qhawaspa."
    es = "Hace mucho tiempo había un cóndor. Cada día volaba mirando a las llamas."
    monkeypatch.setattr(art_plan, "_spanish_title", lambda text_es: "El cóndor y las llamas")
    monkeypatch.setattr(art_plan, "_translate", lambda t, s, g: "Kunturwan llamakunawan" if g == "quy_Latn" else "")
    title, title_es = art_plan.fix_titles("Qhawaspa", "Qhawaspa", story, "quy_Latn", es)
    assert (title, title_es) == ("Kunturwan llamakunawan", "El cóndor y las llamas")
    # a good pair from the LLM is kept as it is
    assert art_plan.fix_titles("Kuntur", "El cóndor", story, "quy_Latn", es) == ("Kuntur", "El cóndor")


def test_fix_titles_spanish_keeps_one_title(monkeypatch):
    monkeypatch.setattr(art_plan, "_spanish_title", lambda text_es: "El zorro y la luna")
    assert art_plan.fix_titles("", "", "El zorro miró la luna.", "spa_Latn", "") == \
        ("El zorro y la luna", "El zorro y la luna")
    assert art_plan.fix_titles("La luna", "x", "El zorro miró la luna.", "spa_Latn", "") == ("La luna", "La luna")


def test_retry_prompt_carries_the_hint():
    s = art_style.emphasize("A condor over the Andes", ["condor"], [], 1,
                            hints={"condor": G.verify_hint("condor")})
    assert s.startswith("Clearly showing condor (an Andean condor: bald head")


def test_verify_asks_with_the_hint(monkeypatch):
    from yq.macworker.models import art_verify, vlm
    seen = {}

    def fake(images, prompt, schema=None, max_tokens=400):
        seen["prompt"] = prompt
        return {"elements": {"condor": True, "llama": True}, "line_art": 9, "text": False, "frame": False}

    monkeypatch.setattr(vlm, "ask_json", fake)
    monkeypatch.setattr(art_verify.config, "mock", lambda name: False)
    v = art_verify.verify(Image.new("RGB", (64, 90), "white"), ["condor", "llama"], culture="andean")
    assert "- condor: an Andean condor" in seen["prompt"] and "- llama: a llama" in seen["prompt"]
    assert v["verified"] == {"condor": True, "llama": True}


def test_vlm_accepts_numpy_images():
    """BOX-ANALYSIS passes numpy RGB arrays; the VLM only took paths/PIL and failed silently."""
    import numpy as np
    from yq.macworker.models.vlm import _prepare
    a = np.zeros((40, 30, 3), np.uint8)
    f = np.full((40, 30, 3), 0.5, np.float32)
    out = _prepare([a, f], max_side=512)
    assert [im.size for im in out] == [(30, 40), (30, 40)] and out[1].getpixel((0, 0)) == (127, 127, 127)
