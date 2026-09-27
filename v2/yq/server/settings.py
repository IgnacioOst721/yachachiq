"""Settings of the kiosk (UI domain). Every value can be changed with YQ_<NAME>.

Shared values (port, idle reset, exit password, consent seconds, folders) live
in yq.common.config; only UI-specific ones are here.
"""
from __future__ import annotations

from pathlib import Path

from yq.common.config import env

HOST = env("KIOSK_HOST", "0.0.0.0")                 # listen on the LAN too (a judge laptop can watch)
# Force the UI's own mocks for these subsystems even when the real module imports.
# "all" or a comma list of: voice, languages, sign, art, box, camera, printer, hologram, publish.
UI_MOCKS = env("UI_MOCKS", [""])
MOCK_SPEED = env("MOCK_SPEED", 1.0)                 # <1 = mocks run faster (tests use 0.02)
PAUSE_SCALE = env("PAUSE_SCALE", 1.0)               # scales the short "read this" pauses (tests use 0.01)

# Portrait/consent camera (the visitor-facing camera; on the robot it is the sign camera).
PORTRAIT_CAMERA = env("PORTRAIT_CAMERA", "0")       # index ("0") or /dev/v4l/by-id/... path
COVERED_BRIGHTNESS = env("COVERED_BRIGHTNESS", 40.0)  # mean grey below this = camera covered = "no"
COVERED_STD = env("COVERED_STD", 12.0)              # or almost uniform image (hand pressed on the lens)

# Kiosk browser (Chromium) that the emergency exit closes.
KIOSK_KILL_PATTERN = env("KIOSK_KILL_PATTERN", "chrom(e|ium).*--kiosk")

# Voice recording limits.
RECORD_MAX_S = env("RECORD_MAX_S", 90.0)

# Timeouts so a hung subsystem never freezes the screen (seconds).
TRANSCRIBE_TIMEOUT = env("TRANSCRIBE_TIMEOUT", 180.0)
TRANSLATE_TIMEOUT = env("TRANSLATE_TIMEOUT", 60.0)
DRAWING_TIMEOUT = env("DRAWING_TIMEOUT", 1200.0)
SCAN_TIMEOUT = env("SCAN_TIMEOUT", 3600.0)
NARRATE_TIMEOUT = env("NARRATE_TIMEOUT", 240.0)
IDLE_WARNING_SECONDS = env("IDLE_WARNING_SECONDS", 10.0)   # "¿Sigues ahí?" shown this long before reset

# How often reachability of the Mac, printer and hologram is refreshed for /status.
STATUS_REFRESH_S = env("STATUS_REFRESH_S", 10.0)

# Paper for the printer job (A4 portrait by default).
PAPER_W_MM = env("PAPER_W_MM", 210.0)
PAPER_H_MM = env("PAPER_H_MM", 297.0)
PENS = env("PENS", ["black"])

STATIC_DIR = Path(__file__).resolve().parent / "static"


def ui_mocked(subsystem: str) -> bool:
    """True when the UI must use its own mock for `subsystem` (tests, demos)."""
    wanted = [s.strip().lower() for s in (UI_MOCKS or []) if s and s.strip()]
    return "all" in wanted or subsystem.lower() in wanted
