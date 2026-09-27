#!/usr/bin/env python3
"""Write the files that make the Jetson start the kiosk by itself (see docs/ui.md).

    python3 tools/ui_install_jetson.py --out ~/yq-install          # writes the files and prints the commands
    python3 tools/ui_install_jetson.py --out ~/yq-install --user ignacio --repo ~/yachachiq

Creates:
  yachachiq-kiosk.service   systemd service for the server (starts at boot, restarts if it crashes)
  yachachiq-kiosk.desktop   desktop autostart entry that opens Chromium full screen after login
It never runs sudo itself: it prints the exact commands to copy the files into place.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

SERVICE = """[Unit]
Description=Yachachiq v2 kiosk server (UI)
# No network dependency on purpose: the robot works offline; publishing is store-and-forward.
After=sound.target

[Service]
Type=simple
User={user}
WorkingDirectory={v2}
ExecStart={v2}/.venvs/ui/bin/python -m yq.server.app
Environment=OPENCV_LOG_LEVEL=FATAL
Environment=YQ_ROLE=jetson
# Environment=YQ_PRINTER_URL=http://192.168.8.30:8900
# Environment=YQ_HOLOGRAM_URL=http://192.168.8.40:8950
# Environment=YQ_KIOSK_EXIT_PASSWORD=ostra
# Environment=YQ_PORTRAIT_CAMERA=/dev/v4l/by-id/usb-XXXX-video-index0
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""

DESKTOP = """[Desktop Entry]
Type=Application
Name=Yachachiq kiosk
Comment=Opens the Yachachiq screen full screen
Exec=/usr/bin/python3 {v2}/tools/ui_kiosk.py
X-GNOME-Autostart-enabled=true
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write systemd + autostart files for the Yachachiq kiosk")
    ap.add_argument("--out", default=str(Path.home() / "yq-install"))
    ap.add_argument("--user", default=getpass.getuser())
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args(argv)
    v2 = Path(os.path.expanduser(a.repo)).resolve() / "v2"
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "yachachiq-kiosk.service").write_text(SERVICE.format(user=a.user, v2=v2))
    (out / "yachachiq-kiosk.desktop").write_text(DESKTOP.format(v2=v2))
    print("Files written in %s. Now run:\n" % out)
    print("  sudo cp %s/yachachiq-kiosk.service /etc/systemd/system/" % out)
    print("  sudo systemctl daemon-reload && sudo systemctl enable --now yachachiq-kiosk")
    print("  mkdir -p ~/.config/autostart && cp %s/yachachiq-kiosk.desktop ~/.config/autostart/" % out)
    print("\nAlso turn on automatic login for user '%s' (Settings > Users) and turn off screen blanking." % a.user)
    return 0


if __name__ == "__main__":
    sys.exit(main())
