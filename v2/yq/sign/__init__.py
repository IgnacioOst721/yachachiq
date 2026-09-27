"""Sign language input for Yachachiq v2 (Jetson side). See docs/sign.md and CONTRACTS.md §6.

    from yq.sign import available, SignEngine
"""
from __future__ import annotations

import json

LANGS = {
    "prl": "Lengua de Señas Peruana (LSP)",
    "ase": "Lengua de Señas Americana (ASL)",
    "ils": "Señas Internacionales",
}


def available() -> list:
    """-> [{"code","name_es","letters": bool,"words": int}] for every sign language, from the
    models actually present on disk (MODELS_DIR/sign or the versioned yq/sign/models)."""
    from . import modelstore
    from .letters import letter_model_lang
    out = []
    for code, name in LANGS.items():
        letters = modelstore.find_model("letters_%s.npz" % letter_model_lang(code)) is not None
        words = 0
        meta = modelstore.find_model("words_%s.json" % code)
        if meta is not None and modelstore.find_model("words_%s.onnx" % code) is not None:
            try:
                words = len(json.loads(meta.read_text()).get("classes", []))
            except (OSError, ValueError):
                words = 0
        out.append({"code": code, "name_es": name, "letters": letters, "words": words})
    return out


def __getattr__(name):          # lazy: importing yq.sign stays cheap
    if name == "SignEngine":
        from .engine import SignEngine
        return SignEngine
    raise AttributeError(name)


__all__ = ["available", "SignEngine", "LANGS"]
