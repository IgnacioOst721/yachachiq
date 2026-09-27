"""UVC camera controls through `v4l2-ctl` (package v4l-utils on the Jetson).

Control names changed in Linux 5.x (exposure_auto -> auto_exposure,
exposure_absolute -> exposure_time_absolute, focus_auto ->
focus_automatic_continuous, white_balance_temperature_auto ->
white_balance_automatic), so we read `v4l2-ctl --list-ctrls-menus` first and
use whichever name the camera reports, clamping every value to its range.
UVC exposure_time_absolute is in units of 100 us (Arducam UVC docs); the
IMX519 USB bridge limits it to 2000 (= 200 ms).
Pure functions here (no subprocess) so the command generation is testable.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from typing import Optional

ALIASES = {
    "auto_exposure": ["auto_exposure", "exposure_auto"],
    "exposure": ["exposure_time_absolute", "exposure_absolute"],
    "exposure_dynamic_framerate": ["exposure_dynamic_framerate", "exposure_auto_priority"],
    "autofocus": ["focus_automatic_continuous", "focus_auto"],
    "focus": ["focus_absolute"],
    "auto_wb": ["white_balance_automatic", "white_balance_temperature_auto"],
    "wb": ["white_balance_temperature"],
    "gain": ["gain"],
}
V4L2_EXPOSURE_MANUAL = 1          # menu value of auto_exposure for "Manual Mode"

_LINE = re.compile(r"^\s*(?P<name>[a-z0-9_]+)\s+0x[0-9a-f]+\s+\((?P<type>[a-z0-9 ]+)\)\s*:\s*(?P<rest>.*)$")
_KV = re.compile(r"(min|max|step|default|value|flags)=(\S+)")
_MENU = re.compile(r"^\s*(\d+):\s*(.+)$")


def parse_ctrls(text: str) -> dict:
    """Parse `v4l2-ctl -d DEV --list-ctrls-menus` into {name: {type,min,max,step,default,value,flags,menu}}."""
    out: dict = {}
    last = None
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            info = {"type": m.group("type").strip(), "menu": {}}
            for k, v in _KV.findall(m.group("rest")):
                info[k] = v if k == "flags" else int(v)
            out[m.group("name")] = last = info
            continue
        mm = _MENU.match(line)
        if mm and last is not None and last["type"] == "menu":
            last["menu"][int(mm.group(1))] = mm.group(2).strip()
    return out


def resolve(ctrls: dict, logical: str) -> Optional[str]:
    for name in ALIASES[logical]:
        if name in ctrls:
            return name
    return None


def clamp(ctrls: dict, name: str, value: int, warnings: list) -> int:
    c = ctrls[name]
    v = int(round(value))
    lo, hi, step = c.get("min"), c.get("max"), c.get("step") or 1
    if lo is not None and hi is not None:
        if v < lo or v > hi:
            warnings.append("%s=%d fuera de rango [%d, %d]" % (name, v, lo, hi))
        v = min(max(v, lo), hi)
        v = lo + ((v - lo) // step) * step
    return v


def lock_plan(ctrls: dict, focus: Optional[int], exposure_us: Optional[int], gain: Optional[int],
              wb_k: Optional[int]) -> tuple:
    """Returns ([stage1 {name: value}, stage2 {name: value}], applied {logical: value}, warnings).
    Stage 1 switches the automatic modes off; stage 2 sets the fixed values
    (manual controls are "inactive" while their auto mode is on)."""
    warnings: list = []
    s1, s2, applied = {}, {}, {}
    name = resolve(ctrls, "auto_exposure")
    if name:
        menu = ctrls[name].get("menu") or {}
        manual = next((k for k, v in menu.items() if "manual" in v.lower()), V4L2_EXPOSURE_MANUAL)
        s1[name] = manual
    elif exposure_us is not None:
        warnings.append("la cámara no tiene auto_exposure")
    for logical in ("autofocus", "auto_wb", "exposure_dynamic_framerate"):
        n = resolve(ctrls, logical)
        if n:
            s1[n] = 0
    for logical, value in (("exposure", None if exposure_us is None else exposure_us / 100.0),
                           ("focus", focus), ("gain", gain), ("wb", wb_k)):
        if value is None:
            continue
        n = resolve(ctrls, logical)
        if not n:
            warnings.append("la cámara no tiene el control %s" % logical)
            continue
        s2[n] = clamp(ctrls, n, value, warnings)
        applied[logical] = s2[n] * 100 if logical == "exposure" else s2[n]
    return [s1, s2], applied, warnings


def set_args(device: str, values: dict) -> list:
    """argv for one v4l2-ctl call setting `values`."""
    if not values:
        return []
    return ["v4l2-ctl", "-d", device, "-c", ",".join("%s=%d" % kv for kv in values.items())]


def get_args(device: str, names) -> list:
    return ["v4l2-ctl", "-d", device, "-C", ",".join(names)]


def parse_get(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = int(v.strip().split()[0])
            except (ValueError, IndexError):
                pass
    return out


def run(argv: list, timeout: float = 5.0) -> str:
    if not shutil.which("v4l2-ctl"):
        raise RuntimeError("v4l2-ctl no está instalado (sudo apt install v4l-utils)")
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("%s -> %s" % (" ".join(argv), (r.stderr or r.stdout).strip()[:300]))
    return r.stdout


def lock_camera(device: str, focus=None, exposure_us=None, gain=None, wb_k=None) -> dict:
    """Apply the plan on a real device and read the values back. Returns
    {"applied": {...}, "readback": {...}, "warnings": [...]}."""
    ctrls = parse_ctrls(run(["v4l2-ctl", "-d", device, "--list-ctrls-menus"]))
    stages, applied, warnings = lock_plan(ctrls, focus, exposure_us, gain, wb_k)
    for values in stages:
        if values:
            run(set_args(device, values))
    names = [n for st in stages for n in st]
    back = parse_get(run(get_args(device, names))) if names else {}
    for st in stages:
        for n, v in st.items():
            if n in back and back[n] != v:
                warnings.append("%s quedó en %d (pedido %d)" % (n, back[n], v))
    return {"applied": applied, "readback": back, "warnings": warnings}
