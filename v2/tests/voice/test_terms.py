"""Andean term post-editing, the garbage guard and the Southern Quechua proxy (bugs found on
2026-09-27: llamas -> flames, kuntur -> águila, an invented Bible verse for Awajún)."""
import pytest

from yq.macworker.models import voice_terms as T
from yq.macworker.models import voice_translate as VT


def test_llamas_are_animals_not_flames():
    src = "El cóndor vuela sobre las montañas y cuida a las llamas del pueblo."
    assert T.postedit(src, "spa_Latn", "The condor watches over the flames of the village.", "eng_Latn") == \
        "The condor watches over the llamas of the village."
    assert T.postedit(src, "spa_Latn", "Kunturqa ninakunata qhawan.", "quy_Latn") == "Kunturqa llamakunata qhawan."


def test_real_fire_keeps_its_flames():
    src = "Las llamas del fuego subían hasta el cielo."
    assert T.postedit(src, "spa_Latn", "The flames of the fire rose to the sky.", "eng_Latn") == \
        "The flames of the fire rose to the sky."


def test_se_llama_is_a_verb():
    src = "La niña se llama Rosa."
    assert T.postedit(src, "spa_Latn", "The girl's name is Rosa, like a flame.", "eng_Latn") == \
        "The girl's name is Rosa, like a flame."


def test_kuntur_is_a_condor():
    src = "Ñawpa pachapi huk kunturmi karqan."
    assert T.postedit(src, "quy_Latn", "En el mundo antiguo, había un águila.", "spa_Latn") == \
        "En el mundo antiguo, había un cóndor."
    # a story with both birds keeps both
    assert T.postedit("El águila y el cóndor volaban.", "spa_Latn", "The eagle and the condor flew.", "eng_Latn") \
        == "The eagle and the condor flew."


def test_garbage_guard():
    assert T.degenerate("El cóndor cuida a las llamas.", "13:20 Tuja aishnuma, tuja aishnuma. Tuja aishnuma, tuja aishnuma.")
    assert T.degenerate("Hola.", "")
    assert T.degenerate("Una historia larga sobre un cóndor que vuela.", "Sí.")
    assert not T.degenerate("El cóndor cuida a las llamas.", "The condor watches over the llamas.")
    assert not T.degenerate("Juan 3:16 dice algo.", "John 3:16 says something.")


class FakeModel:
    def __init__(self, out):
        self.out, self.calls = out, []

    def translate_batch(self, pieces, src_code, tgt_code, beam=4):
        self.calls.append((src_code, tgt_code))
        return [self.out for _ in pieces]


def test_cusco_quechua_goes_through_nllb_as_ayacucho():
    m = FakeModel("En el mundo antiguo, había un águila.")
    d = VT.translate_detail("Ñawpa pachapi huk kunturmi karqan.", "quz_Latn", "spa_Latn", get_model=lambda e: m)
    assert d["engine"] == "nllb" and d["verified"] and d["via"] == "quy_Latn"
    assert m.calls == [("quy_Latn", "spa_Latn")]
    assert d["text"] == "En el mundo antiguo, había un cóndor."


def test_garbage_translation_is_refused():
    m = FakeModel("13:20 Tuja aishnuma, tuja aishnuma. Tuja aishnuma, tuja aishnuma.")
    with pytest.raises(ValueError, match="garbage"):
        VT.translate_detail("El cóndor cuida a las llamas del pueblo.", "spa_Latn", "agr_Latn",
                            engine="madlad", get_model=lambda e: m)
