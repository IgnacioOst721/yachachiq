"""Finding the box ESP32's serial port on the Jetson (Linux) or a Mac."""
from __future__ import annotations

import glob
import os
import sys
from typing import Optional

from . import settings as S


def _matches(name: str, hints) -> bool:
    low = name.lower()
    return any(h.lower() in low for h in hints)


def candidate_ports(platform: Optional[str] = None, hints=None) -> list:
    """Likely ESP32 ports, best first.

    Linux: /dev/serial/by-id/* (stable names that include the USB-UART chip,
    e.g. usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_<serial>-if00-port0),
    then /dev/ttyUSB*, /dev/ttyACM*. macOS: /dev/cu.* (never /dev/tty.*, which
    blocks on open waiting for carrier detect), skipping Bluetooth ports.
    """
    platform = platform or sys.platform
    hints = hints or S.SERIAL_HINTS
    out: list = []
    if platform.startswith("linux"):
        by_id = sorted(glob.glob("/dev/serial/by-id/*"))
        out += [p for p in by_id if _matches(os.path.basename(p), hints)]
        out += [p for p in by_id if p not in out]
        out += sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    elif platform == "darwin":
        cu = [p for p in sorted(glob.glob("/dev/cu.*")) if "bluetooth" not in p.lower() and "debug-console" not in p.lower()]
        out += [p for p in cu if _matches(os.path.basename(p), hints)]
        out += [p for p in cu if p not in out and ("usb" in p.lower())]
    else:
        out += sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    seen, uniq = set(), []
    for p in out:  # by-id links and ttyUSB0 can be the same device
        real = os.path.realpath(p)
        if real not in seen:
            seen.add(real)
            uniq.append(p)
    return uniq
