"""Run the REAL box firmware in Espressif's QEMU and test its protocol with the
REAL Python driver (yq.box.device.Box) over a pseudo-terminal.

QEMU (github.com/espressif/qemu) emulates the ESP32 CPU, flash, NVS and UART,
but not the MCPWM/PCNT peripherals of the stepper engine, so this uses the
`qemu` PlatformIO environment (same firmware, -DYQ_NO_STEPPER). GPIO inputs read
LOW in QEMU: both reeds look "closed" and the HX711 looks always ready (raw 0).

    ~/.local/bin/pio run -d v2/firmware/box_esp32 -e qemu
    YQ_BOX_QEMU=/path/to/qemu-system-xtensa .venvs/box/bin/python tools/box_qemu.py

On macOS the Espressif build links Homebrew's pixman/libgcrypt/SDL2; without
Homebrew point DYLD_FALLBACK_LIBRARY_PATH at a folder with those dylibs.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
FW = V2 / "firmware" / "box_esp32"
BUILD = FW / ".pio" / "build" / "qemu"
ESPTOOL = Path.home() / ".platformio" / "packages" / "tool-esptoolpy" / "esptool.py"
BOOT_APP0 = Path.home() / ".platformio" / "packages" / "framework-arduinoespressif32" / "tools" / "partitions" / "boot_app0.bin"
PIO_PY = Path.home() / ".local" / "share" / "uv" / "tools" / "platformio" / "bin" / "python"


def merged_image(out: Path) -> Path:
    subprocess.run([str(PIO_PY), str(ESPTOOL), "--chip", "esp32", "merge_bin", "--fill-flash-size", "4MB",
                    "-o", str(out), "0x1000", str(BUILD / "bootloader.bin"), "0x8000", str(BUILD / "partitions.bin"),
                    "0xe000", str(BOOT_APP0), "0x10000", str(BUILD / "firmware.bin")],
                   check=True, capture_output=True)
    return out


def start_qemu(image: Path):
    qemu = os.environ.get("YQ_BOX_QEMU", "qemu-system-xtensa")
    proc = subprocess.Popen([qemu, "-nographic", "-machine", "esp32", "-drive", "file=%s,if=mtd,format=raw" % image,
                             "-serial", "pty", "-monitor", "none"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    deadline = time.time() + 20
    while time.time() < deadline:
        line = proc.stdout.readline()
        m = re.search(r"char device redirected to (\S+)", line or "")
        if m:
            return proc, m.group(1)
    proc.kill()
    raise RuntimeError("QEMU did not open a pty")


def run_checks(port: str) -> list:
    from yq.box.device import Box
    from yq.box.protocol import BoxError
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print("%-4s %-46s %s" % ("OK" if ok else "FAIL", name, detail))

    def err(fn):
        try:
            fn()
        except BoxError as e:
            return e
        return None

    box = Box(port=port, heartbeat=True).connect(timeout=15)
    boot = box.wait_event("boot", timeout=10)
    check("boot event", boot.get("fw") == "yq-box", str(boot))
    info = box.info()
    check("info: firmware + protocol", info["fw"] == "yq-box" and info["proto"] == 1, info["version"])
    check("info: 12 channels, 44800 steps/rev", len(info["channels"]) == 12 and info["steps_per_rev"] == 44800)
    check("TMC2209 absent is reported", info["tmc"]["ok"] is False)
    time.sleep(0.2)
    st = box.status()
    check("status: doors closed (reeds LOW)", st["doors_closed"] is True, str(st["doors"]))
    check("status: fault list", "tmc_uart" in st["faults"], str(st["faults"]))
    e = err(lambda: box.rotate_to(10))
    check("motion refused without TMC UART", e is not None and e.code == "tmc_uart")
    r = box.light("rake1", 1.0, max_ms=300)
    check("light with max_ms", r["off_in_ms"] == 300, str(r))
    ev = box.wait_event("light_off", lambda m: m["ch"] == "rake1", timeout=2)
    check("auto switch-off at max_ms", ev["reason"] == "max_on" and 290 <= ev["on_ms"] <= 330, str(ev))
    e = err(lambda: box.light("rake1", 1.0))
    wait = (e.reply or {}).get("wait_ms", 0) if e else 0
    check("cool-down 2x on-time", e is not None and e.code == "cooldown" and 450 <= wait <= 620, "wait_ms=%s" % wait)
    box.light("rake2", 1.0, max_ms=2000)
    box.light("rake3", 1.0, max_ms=2000)
    ev = box.wait_event("light_off", lambda m: m["ch"] == "rake2", timeout=2)
    check("only one raking LED", ev["reason"] == "exclusive")
    r = box.light("uv", 0.3, max_ms=500)
    check("UV on with doors closed (on/off only)", r["level"] == 1.0, str(r))
    box.all_off()
    cfg = box.config_set(max_on_ms={"halogen": 999999}, cool_factor={"halogen": 0.1}, hb_timeout_ms=700)
    check("hard caps enforced", cfg["max_on_ms"]["halogen"] == 60000 and cfg["cool_factor"]["halogen"] == 2.0,
          str(cfg["max_on_ms"]))
    r = box.request("config_set", microsteps=32)
    check("drive geometry needs a reboot", r.get("reboot_required") is True and r["config"]["microsteps"] == 16)
    box.request("config_set", microsteps=16)
    e = err(lambda: box.request("config_get", bogus=1))
    check("unknown keys ignored", e is None)
    from yq.box.device import _Pending
    for rid, line, code in ((-1, b"{broken\n", "bad_json"), (90005, b'{"id":90005,"cmd":"fly"}\n', "unknown_cmd"),
                            (-1, b'{"id":6,"cmd":"ping","pad":"' + b"x" * 600 + b'"}\n', "line_too_long")):
        slot = _Pending()
        with box._plock:
            box._pending[rid] = slot
        box._write(line)
        got = slot.event.wait(3) and slot.msg or {}
        with box._plock:
            box._pending.pop(rid, None)
        check("error %s" % code, got.get("error") == code, str(got))
    tare = box.tare(n=5, timeout_ms=2000)
    check("tare (HX711 input stuck LOW in QEMU)", tare["offset"] == 0 and tare["stable"], str(tare))
    e = err(lambda: box.weigh(n=5, timeout_ms=2000))
    check("weigh refused before calibration", e is not None and e.code == "not_calibrated")
    e = err(lambda: box.calibrate_scale(100.0, n=5, timeout_ms=2000))
    check("calibration refuses a missing weight", e is not None and e.code == "bad_param")
    box.config_set(save=True, max_on_ms={"rake": 12000}, hb_timeout_ms=700)
    box.request("reboot")
    box.wait_event("boot", lambda m: m.get("reset_reason") == "software", timeout=15)
    check("NVS keeps config after reboot", box.config_get()["max_on_ms"]["rake"] == 12000)
    box.heartbeat_enabled = False
    box.light("fan", 1.0)
    t0 = time.time()
    ev = box.wait_event("watchdog", timeout=5)
    dt = time.time() - t0
    check("heartbeat watchdog (700 ms)", ev["reason"] == "heartbeat" and 0.6 < dt < 1.5, "%.2f s" % dt)
    st = box.status()
    check("watchdog turned the fan off", st["lights"]["fan"]["level"] == 0)
    box.close()
    return results


def main() -> int:
    if not (BUILD / "firmware.bin").is_file():
        print("build first: pio run -d %s -e qemu" % FW)
        return 2
    with tempfile.TemporaryDirectory() as td:
        image = merged_image(Path(td) / "flash.bin")
        proc, port = start_qemu(image)
        try:
            results = run_checks(port)
        finally:
            proc.kill()
    failed = [r for r in results if not r[1]]
    print("%d checks, %d failed" % (len(results), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
