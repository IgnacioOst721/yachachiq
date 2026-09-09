"""GRBL sender for the Yachachiq pen plotter (Arduino UNO + CNC Shield, grbl-servo).

Send one line, wait for GRBL's "ok" (simple and safe at plotter speeds), report
progress, allow a clean stop, and wait until the machine is really idle at the
end (GRBL says "ok" when a line enters its buffer, not when the pen stops).

Mock mode (no pyserial, no port, or config.MOCK): lines go to `self.sent`,
with a small delay so the UI animates. `mode` tells you which one you got.
"""
import glob
import logging
import os
import re
import threading
import time

try:
    import serial
except ImportError:              # pragma: no cover
    serial = None

import config
import gcode as gcode_mod

log = logging.getLogger("plotter")


def _find_port(port):
    if port and os.path.exists(port):
        return port
    for pat in ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/cu.usbmodem*", "/dev/cu.usbserial*"):
        found = sorted(glob.glob(pat))
        if found:
            return found[0]
    return None


def _clean(line):
    s = line.strip()
    if not s or s.startswith("(") or s.startswith(";"):
        return ""
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    return s


class Plotter:
    def __init__(self, port=None, baud=None, mock=None):
        self.port = port or config.SERIAL_PORT
        self.baud = baud or config.BAUD_RATE
        self.mock = config.MOCK if mock is None else mock
        self.mode = "mock" if self.mock else "unknown"
        self.ser = None
        self.sent = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_status = ""
        self.missing = False               # True when we are mock only because no board was found

    # --- connection ------------------------------------------------------------------------------
    def connect(self):
        self._stop.clear()
        if self.mock:
            self.mode = "mock"
            return self
        port = _find_port(self.port)
        if serial is None or port is None:
            log.warning("plotter: %s -> mock", "pyserial missing" if serial is None else f"no port ({self.port})")
            self.mock, self.mode = True, "mock"
            self.missing = serial is not None          # a board may still be plugged in later
            return self
        try:
            self.ser = serial.Serial(port, self.baud, timeout=2)
            self.ser.write(b"\r\n\r\n")
            time.sleep(2)
            self.ser.reset_input_buffer()
            self.port, self.mode, self.missing = port, "real", False
            log.info("plotter connected on %s", port)
        except Exception as e:
            log.warning("plotter: could not open %s (%s) -> mock", port, e)
            self.ser, self.mock, self.mode = None, True, "mock"
        return self

    def reconnect_if_needed(self):
        """The board was not there when the robot started (or the cable was pulled): try again.
        Lets the Arduino be plugged in after power-on and still get a real drawing."""
        if config.MOCK:
            return False
        if self.ser is not None:
            try:
                if os.path.exists(self.port):
                    return True
            except Exception:
                pass
            log.warning("plotter: %s disappeared, reconnecting", self.port)
            self.close()
            self.mock, self.mode, self.missing = True, "mock", True
        if not getattr(self, "missing", True) and self.ser is not None:
            return True
        if _find_port(config.SERIAL_PORT) is None:
            return False
        self.mock, self.port = False, config.SERIAL_PORT
        self.connect()
        return self.mode == "real"

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            finally:
                self.ser = None

    # --- low level ----------------------------------------------------------------------------------
    def _readline(self):
        if self.ser is None:
            return ""
        return self.ser.readline().decode(errors="ignore").strip()

    def send(self, line):
        """Send one line and block until GRBL answers ok/error. Returns the reply."""
        clean = _clean(line)
        if not clean:
            return "ok"
        with self._lock:
            self.sent.append(clean)
            if self.mock:
                if gcode_mod.is_motion(clean):
                    time.sleep(config.MOCK_LINE_DELAY)
                return "ok"
            self.ser.write((clean + "\n").encode())
            t0 = time.time()
            while time.time() - t0 < 120:
                resp = self._readline()
                if not resp:
                    continue
                low = resp.lower()
                if low.startswith("ok"):
                    return resp
                if low.startswith("error") or low.startswith("alarm"):
                    log.warning("GRBL %s on: %s", resp, clean)
                    return resp
                if resp.startswith("<"):
                    self.last_status = resp
            return "timeout"

    def status(self):
        """Ask GRBL for a status report; returns e.g. 'Idle', 'Run', 'Alarm', or '' in mock."""
        if self.mock or self.ser is None:
            return "Idle"
        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write(b"?")
            time.sleep(0.15)
            data = self.ser.read(self.ser.in_waiting or 1).decode(errors="ignore")
        m = re.search(r"<([A-Za-z]+)", data)
        if m:
            self.last_status = data.strip()
            return m.group(1)
        return ""

    def wait_idle(self, timeout=600):
        """Block until the machine has really finished moving."""
        if self.mock:
            return True
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._stop.is_set():          # the drawing was cancelled: nothing to wait for
                return False
            if self.status() == "Idle":
                return True
            time.sleep(0.3)
        log.warning("plotter: still not idle after %ss", timeout)
        return False

    # --- high level -----------------------------------------------------------------------------------
    def unlock(self):
        return self.send("$X")

    def soft_reset(self):
        if self.mock or self.ser is None:
            return
        with self._lock:
            self.ser.write(b"\x18")
            time.sleep(2)
            self.ser.reset_input_buffer()

    def stop(self):
        self._stop.set()
        self.soft_reset()

    def run(self, lines, on_progress=None):
        """Stream lines. on_progress(sent, total). Returns True if it finished."""
        self._stop.clear()
        motion = [l for l in lines if _clean(l)]
        total = len(motion)
        for i, line in enumerate(motion, 1):
            if self._stop.is_set():
                log.info("plotter run stopped at %d/%d", i, total)
                return False
            self.send(line)
            if on_progress and (i % 5 == 0 or i == total):
                on_progress(i, total)
        self.wait_idle()
        return True

    def stream_file(self, path, on_progress=None):
        with open(path) as f:
            return self.run(f.readlines(), on_progress)

    # --- manual control (gear menu on the screen) --------------------------------------------------------
    def jog(self, dx=0.0, dy=0.0, dz=0.0, feed=None):
        feed = feed or config.TRAVEL_FEED
        parts = []
        if dx: parts.append(f"X{dx:.2f}")
        if dy: parts.append(f"Y{dy:.2f}")
        if dz: parts.append(f"Z{dz:.2f}")
        if not parts:
            return
        self.send("G91")
        self.send("G1 " + " ".join(parts) + f" F{int(feed if not dz else config.PEN_FEED)}")
        self.send("G90")

    def pen_up(self):
        for l in gcode_mod.pen_up():
            self.send(l)

    def pen_down(self):
        for l in gcode_mod.pen_down():
            self.send(l)

    def set_origin(self):
        """Call with the pen at the bottom-left corner of the paper, touching it."""
        return self.send("G92 X0 Y0 Z0")

    def home_xy(self):
        self.pen_up()
        self.send("G0 X0 Y0")

    def load_settings(self, path):
        with open(path) as f:
            for line in f:
                if line.strip().startswith("$"):
                    self.send(line)


def create_plotter(mock=None):
    return Plotter(mock=mock)
