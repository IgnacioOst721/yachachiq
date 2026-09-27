import os
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

import pytest


@pytest.fixture(autouse=True)
def _isolated_data(tmp_path, monkeypatch):
    """Every test writes into its own temp data folder, never ~/yq-data."""
    from yq.common import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    for name in ("STORIES_DIR", "SCANS_DIR", "CALIB_DIR", "MODELS_DIR", "CATALOG_DIR", "JOBS_DIR", "LOG_DIR"):
        monkeypatch.setattr(config, name, tmp_path / "data" / name.lower().replace("_dir", ""))
    config.ensure_dirs()
    yield
