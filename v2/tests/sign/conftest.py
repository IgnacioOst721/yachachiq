from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def real_models(monkeypatch):
    """Heavy tests read the downloaded pose models from the real MODELS_DIR (not the temp one)."""
    from yq.common import config
    real = Path(os.environ.get("YQ_MODELS_DIR", Path.home() / "yq-data" / "models")).expanduser()
    monkeypatch.setattr(config, "MODELS_DIR", real)
    return real
