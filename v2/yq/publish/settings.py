"""Publishing settings (YQ_<NAME> env vars)."""
from __future__ import annotations

from pathlib import Path

from yq.common.config import env

PUBLISH_ENABLED = env("PUBLISH_ENABLED", True)
# Separate clone of the gallery repo's `main` branch (GitHub Pages serves main:/docs).
PUBLISH_REPO_DIR = env("PUBLISH_REPO_DIR", Path.home() / "yachachiq-archivo")
PUBLISH_REMOTE = env("PUBLISH_REMOTE", "origin")
PUBLISH_BRANCH = env("PUBLISH_BRANCH", "main")
PUBLISH_EVERY_S = env("PUBLISH_EVERY_S", 300.0)        # background retry while stories are pending


def ui_mocked() -> bool:
    from yq.server import settings as ui
    return ui.ui_mocked("publish")
