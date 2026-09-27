"""Yachachiq v2 settings, shared by the Jetson and the MacBook.

Every value can be overridden with an environment variable named YQ_<NAME>
(example: YQ_MAC_HOST=192.168.8.20). YQ_MOCK=1 turns every subsystem into a
simulator so the whole robot runs on any computer with no hardware and no
models; YQ_MOCK_<SUBSYSTEM>=1 mocks only one of them (BOX, CAMERAS, THERMAL,
MIC, SPEAKER, SIGN_CAMERA, MAC, PRINTER, HOLOGRAM).

Python 3.10 compatible: the Jetson runs JetPack 6 (Ubuntu 22.04, Python 3.10).
"""
from __future__ import annotations

import os
from pathlib import Path

V2_DIR = Path(__file__).resolve().parents[2]      # .../yachachiq/v2
REPO_DIR = V2_DIR.parent


def env(name: str, default):
    """Read YQ_<name> from the environment, typed like `default`."""
    raw = os.environ.get("YQ_" + name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, (list, tuple)):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(default, Path):
        return Path(raw).expanduser()
    return raw


# --- Role and mocks -----------------------------------------------------------
ROLE = env("ROLE", "jetson")          # "jetson" | "mac" | "dev"
MOCK = env("MOCK", False)


def mock(subsystem: str) -> bool:
    """True when `subsystem` must be simulated (global YQ_MOCK or YQ_MOCK_<SUBSYSTEM>)."""
    return MOCK or env("MOCK_" + subsystem.upper(), False)


# --- Folders ------------------------------------------------------------------
DATA_DIR = env("DATA_DIR", Path.home() / "yq-data")
STORIES_DIR = env("STORIES_DIR", DATA_DIR / "stories")   # one folder per story
SCANS_DIR = env("SCANS_DIR", DATA_DIR / "scans")         # one folder per box scan (CONTRACTS.md §4)
CALIB_DIR = env("CALIB_DIR", DATA_DIR / "calibration")   # camera intrinsics, RTI lights, thermal registration
MODELS_DIR = env("MODELS_DIR", DATA_DIR / "models")      # our own trained models (sign language, etc.)
CATALOG_DIR = env("CATALOG_DIR", DATA_DIR / "catalog")   # offline museum reference catalog (Mac)
JOBS_DIR = env("JOBS_DIR", DATA_DIR / "jobs")            # Mac worker job folders
LOG_DIR = env("LOG_DIR", DATA_DIR / "logs")

# --- Network (GL.iNet Beryl AX, default LAN 192.168.8.0/24) --------------------
# Give the machines fixed DHCP leases in the router so the fallback IPs always work.
MAC_HOST = env("MAC_HOST", "yachachiq-mac.local")
MAC_FALLBACK_IP = env("MAC_FALLBACK_IP", "192.168.8.20")
MAC_PORT = env("MAC_PORT", 8700)
JETSON_HOST = env("JETSON_HOST", "yachachiq-jetson.local")
JETSON_FALLBACK_IP = env("JETSON_FALLBACK_IP", "192.168.8.10")
JETSON_PORT = env("JETSON_PORT", 8877)
PRINTER_URL = env("PRINTER_URL", "http://192.168.8.30:8900")    # Joaquín's printer (interface TBD with him)
HOLOGRAM_URL = env("HOLOGRAM_URL", "http://192.168.8.40:8950")  # Joaquín's hologram
PUBLIC_BASE_URL = env("PUBLIC_BASE_URL", "https://ignacioost721.github.io/yachachiq/")  # QR target (web gallery)
HTTP_TIMEOUT = env("HTTP_TIMEOUT", 10.0)          # short calls
JOB_TIMEOUT = env("JOB_TIMEOUT", 900.0)           # long Mac jobs (image, reconstruction, identification)


def mac_urls() -> list[str]:
    """Ways to reach the Mac worker, best first: mDNS name, then the fixed IP."""
    out: list[str] = []
    for host in (MAC_HOST, MAC_FALLBACK_IP):
        if host:
            url = "http://%s:%d" % (host, MAC_PORT)
            if url not in out:
                out.append(url)
    return out


# --- Mac worker memory budget (MacBook M4, 16 GB unified memory) ----------------
# Models are loaded on demand and evicted least-recently-used above this budget,
# leaving room for macOS and the reconstruction jobs.
MAC_MODEL_BUDGET_GB = env("MAC_MODEL_BUDGET_GB", 10.5)

# --- UI -------------------------------------------------------------------------
UI_LANGUAGE = env("UI_LANGUAGE", "spa_Latn")      # kiosk text language (canonical code)
IDLE_RESET_SECONDS = env("IDLE_RESET_SECONDS", 45.0)
KIOSK_EXIT_PASSWORD = env("KIOSK_EXIT_PASSWORD", "ostra")
CONSENT_SECONDS = env("CONSENT_SECONDS", 10)


def ensure_dirs() -> None:
    for d in (DATA_DIR, STORIES_DIR, SCANS_DIR, CALIB_DIR, MODELS_DIR, CATALOG_DIR, JOBS_DIR, LOG_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
