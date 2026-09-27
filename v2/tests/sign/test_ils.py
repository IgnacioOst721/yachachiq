"""International Sign uses the ASL letter model (same one-handed manual alphabet) until the team
trains its own; a letters_ils.npz in MODELS_DIR/sign takes over automatically."""
import shutil

from yq.sign import available, letters, modelstore


def test_ils_has_letters_from_asl():
    ils = next(x for x in available() if x["code"] == "ils")
    assert ils["letters"] is True
    assert letters.letter_model_lang("ils") == "ase"
    a, i = letters.LetterClassifier.load("ase"), letters.LetterClassifier.load("ils")
    assert a.classes == i.classes
    assert set(letters.MOTION_LETTERS["ils"]) == {"j", "z"}


def test_a_team_trained_ils_model_wins(tmp_path, monkeypatch):
    own = tmp_path / "sign"
    own.mkdir()
    src = modelstore.find_model("letters_ase.npz")
    shutil.copy(src, own / "letters_ils.npz")
    shutil.copy(src.with_suffix(".json"), own / "letters_ils.json")
    monkeypatch.setattr(modelstore, "sign_dir", lambda: own)
    assert letters.letter_model_lang("ils") == "ils"
