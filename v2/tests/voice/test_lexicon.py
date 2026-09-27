"""Quechua/Aymara story dictionary: NLLB turned 'atuq' (fox) into 'un grupo' and 'qucha' (lake) into
'el mar' (2026-09-27); the planner and the translation post-editor now get the reliable meanings."""
from yq.macworker.models import andean_lexicon as A
from yq.macworker.models import voice_translate as VT

FOX = "Ñawpa pachapi huk atuqmi karqan. Sapa tuta killata qhawaspa qucha patapi takirqan."


def test_roots_match_with_suffixes():
    got = {es.split(",")[0] for _w, _r, es, _en in A.hits(FOX, "quz_Latn")}
    assert {"zorro", "luna", "laguna", "cantar", "noche"} <= got
    assert "anciana" not in got                      # 'payqa' must not match 'paya'
    kuntur = "Payqa sapa p'unchaw hatun urqukunapa hawanta phawarqan, llamakunata qhawaspa."
    assert {"cerro", "volar", "llama", "día"} <= {es.split(",")[0] for *_x, es, _en in A.hits(kuntur, "quy_Latn")}
    assert A.hits("Mä qamaqix phaxsi uñch'ukiwayi", "ayr_Latn")[0][2] == "zorro"
    assert A.hits("El zorro canta", "spa_Latn") == []


def test_llm_postedit_fixes_the_fox(monkeypatch):
    from yq.macworker.models import llm
    seen = {}

    def fake_chat(msgs, max_tokens=800, temperature=0.2, **kw):
        seen["user"] = msgs[-1]["content"]
        return "En tiempos antiguos había un zorro que cada noche miraba la luna y cantaba a orillas de la laguna."

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(A, "_table", A._table)
    import yq.common.config as C
    monkeypatch.setattr(C, "mock", lambda name: False)
    draft = "En el mundo antiguo, había un grupo que veía la luna cada noche y cantaba sobre el mar."
    out = A.llm_postedit(FOX, "quz_Latn", draft, "spa_Latn")
    assert "zorro" in out and "laguna" in out
    assert "atuqmi = zorro" in seen["user"] and "qucha = laguna" in seen["user"]


def test_llm_postedit_rejects_a_rewrite(monkeypatch):
    from yq.macworker.models import llm
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "Sí.")
    import yq.common.config as C
    monkeypatch.setattr(C, "mock", lambda name: False)
    draft = "En el mundo antiguo, había un grupo que veía la luna cada noche y cantaba sobre el mar."
    assert A.llm_postedit(FOX, "quz_Latn", draft, "spa_Latn") == draft


def test_translate_detail_postedits_quechua(monkeypatch):
    class Fake:
        def translate_batch(self, pieces, s, t, beam=4):
            return ["En el mundo antiguo, había un grupo que veía la luna cada noche y cantaba sobre el mar."]
    monkeypatch.setattr(A, "llm_postedit", lambda src_text, src, draft, tgt, found=None:
                        draft.replace("un grupo", "un zorro").replace("el mar", "la laguna"))
    d = VT.translate_detail(FOX, "quz_Latn", "spa_Latn", get_model=lambda e: Fake())
    assert d["postedit"] == "glossary+llm" and "zorro" in d["text"] and "laguna" in d["text"]


def test_planner_gets_the_dictionary():
    from yq.macworker.models import art_plan
    msg = art_plan._user_message(FOX, "quz_Latn", "había un grupo que cantaba sobre el mar", "")
    assert "RELIABLE" in msg and "atuqmi = zorro (fox)" in msg and "qucha = laguna, lago" in msg


def test_synonyms_satisfy_the_check(monkeypatch):
    """'montañas' is fine for urqu: the post-edit must not force the literal word 'cerro'."""
    from yq.macworker.models import llm
    import yq.common.config as C
    monkeypatch.setattr(C, "mock", lambda name: False)
    calls = []
    good = "Había un cóndor que volaba sobre las montañas y miraba a las llamas cada día."
    monkeypatch.setattr(llm, "chat", lambda msgs, **k: (calls.append(1), good)[1])
    src = "Huk kunturmi karqan. Sapa p'unchaw urqukunapa hawanta phawarqan, llamakunata qhawaspa."
    assert A.llm_postedit(src, "quy_Latn", "Había un águila que volaba sobre las montañas.", "spa_Latn") == good
    assert len(calls) == 1                   # accepted at once: no retry asking for 'cerro'
