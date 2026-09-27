#!/usr/bin/env python3
"""Open the Yachachiq kiosk full screen in Chromium (Jetson or any Linux desktop).

    python3 tools/ui_kiosk.py                  # waits for the server, then opens Chromium in kiosk mode
    python3 tools/ui_kiosk.py --url http://localhost:8877 --dry-run

Needs only the Python standard library (it runs from the desktop autostart, outside any venv).
The emergency exit in the gear menu kills this Chromium (pattern "chrom(e|ium).*--kiosk").
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request

FLAGS = [
    "--kiosk", "--noerrdialogs", "--disable-infobars", "--disable-session-crashed-bubble",
    "--check-for-update-interval=31536000", "--touch-events=enabled", "--disable-pinch",
    "--overscroll-history-navigation=0", "--disable-features=Translate,TranslateUI",
    "--lang=es", "--password-store=basic", "--no-first-run", "--start-fullscreen",
]


def browser() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome"):
        path = shutil.which(name)
        if path:
            return path
    snap = "/snap/bin/chromium"
    return snap if os.path.exists(snap) else ""


def wait_for(url: str, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/status", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8877")
    ap.add_argument("--wait", type=float, default=120.0, help="seconds to wait for the server")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not start Chromium")
    a = ap.parse_args(argv)
    exe = browser()
    cmd = [exe or "chromium"] + FLAGS
    if os.environ.get("WAYLAND_DISPLAY"):
        cmd.append("--ozone-platform=wayland")
    cmd.append(a.url)
    if a.dry_run:
        print(" ".join(cmd))
        return 0
    if not exe:
        print("Chromium not found: sudo apt install chromium-browser (or snap install chromium)", file=sys.stderr)
        return 1
    if not wait_for(a.url, a.wait):
        print("the kiosk server did not answer at %s; opening anyway" % a.url, file=sys.stderr)
    os.execv(exe, cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
