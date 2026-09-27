"""JSON-lines protocol shared by the ESP32 firmware, the driver and the simulator.

The authoritative description is firmware/box_esp32/PROTOCOL.md. This module
only encodes/decodes lines and defines the error types; it has no I/O.
"""
from __future__ import annotations

import json
from typing import Optional

MAX_LINE = 512
PROTO_VERSION = 1

COMMANDS = ("ping", "info", "hb", "status", "motor", "rotate_to", "rotate_by", "speed", "stop", "zero",
            "scale_tare", "scale_cal", "weigh", "light", "all_off", "estop", "config_get", "config_set",
            "reboot")
EVENTS = ("boot", "door", "interlock", "light_off", "move_done", "watchdog", "fault")
ERRORS = ("bad_json", "line_too_long", "unknown_cmd", "bad_param", "interlock", "cooldown", "moving",
          "tmc_uart", "scale_busy", "not_tared", "not_calibrated", "scale_timeout", "hx711_missing")
# Safe to re-send after a lost reply (same effect twice). rotate_by is not:
# the driver implements rotate_by with rotate_to on an absolute target instead.
IDEMPOTENT = {"ping", "info", "hb", "status", "motor", "rotate_to", "speed", "stop", "zero", "light",
              "all_off", "estop", "config_get", "config_set"}
SLOW = {"scale_tare", "scale_cal", "weigh"}   # reply when the measurement ends

ERROR_ES = {
    "interlock": "Hay una puerta abierta: la luz UV y la lámpara halógena no pueden encenderse.",
    "cooldown": "Esa luz se está enfriando; espera un momento.",
    "moving": "El plato está girando; espera a que se detenga.",
    "tmc_uart": "El driver del motor (TMC2209) no responde por UART.",
    "scale_busy": "La balanza está midiendo.",
    "not_tared": "Primero hay que tarar la balanza con el plato vacío.",
    "not_calibrated": "La balanza no está calibrada (usa calibrate-scale con un peso conocido).",
    "scale_timeout": "La balanza no se estabilizó (vibración, ventilador o corriente de aire).",
    "hx711_missing": "No se detecta el HX711 de la balanza.",
}


class BoxError(RuntimeError):
    """The ESP32 answered ok:false (code = the protocol error code)."""

    def __init__(self, code: str, msg: str = "", reply: Optional[dict] = None):
        self.code = code
        self.msg = msg
        self.reply = reply or {}
        super().__init__("%s%s" % (code, (": " + msg) if msg else ""))

    @property
    def message_es(self) -> str:
        return ERROR_ES.get(self.code, "Error de la caja: %s" % self)


class BoxTimeout(BoxError):
    def __init__(self, msg: str = ""):
        super().__init__("timeout", msg)


class BoxUnavailable(BoxError):
    def __init__(self, msg: str = ""):
        super().__init__("unavailable", msg)


class ProtocolError(ValueError):
    pass


def encode_request(req_id: int, cmd: str, **params) -> bytes:
    if cmd not in COMMANDS:
        raise ProtocolError("unknown command %r" % cmd)
    msg = {"id": int(req_id), "cmd": cmd}
    for k, v in params.items():
        if v is not None:
            msg[k] = v
    line = json.dumps(msg, separators=(",", ":"), ensure_ascii=True)
    if len(line) + 1 > MAX_LINE:
        raise ProtocolError("request longer than %d bytes" % MAX_LINE)
    return (line + "\n").encode("ascii")


def encode_message(msg: dict) -> bytes:
    """Encode a reply/event (simulator side)."""
    return (json.dumps(msg, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def decode_line(line) -> dict:
    """Parse one line. Returns the message dict with an added "_kind":
    "reply" (has id), "event" (has event) or raises ProtocolError."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line:
        raise ProtocolError("empty line")
    if not line.startswith("{"):
        raise ProtocolError("not JSON: %r" % line[:80])   # e.g. ROM bootloader noise after a reset
    try:
        msg = json.loads(line)
    except ValueError as e:
        raise ProtocolError("bad JSON: %s" % e)
    if not isinstance(msg, dict):
        raise ProtocolError("not an object")
    if "event" in msg:
        msg["_kind"] = "event"
    elif "id" in msg and "ok" in msg:
        msg["_kind"] = "reply"
    else:
        raise ProtocolError("neither reply nor event: %r" % line[:80])
    return msg


def raise_for_reply(msg: dict) -> dict:
    if msg.get("ok"):
        return msg
    raise BoxError(str(msg.get("error") or "error"), str(msg.get("msg") or ""), msg)


class LineBuffer:
    """Accumulates raw bytes and yields complete lines (without the newline)."""

    def __init__(self, max_line: int = 4096):
        self._buf = bytearray()
        self.max_line = max_line
        self.dropped = 0

    def feed(self, data: bytes) -> list:
        self._buf.extend(data)
        lines = []
        while True:
            i = self._buf.find(b"\n")
            if i < 0:
                break
            raw = bytes(self._buf[:i]).rstrip(b"\r")
            del self._buf[: i + 1]
            if raw:
                lines.append(raw)
        if len(self._buf) > self.max_line:   # garbage without newline (baud mismatch, boot ROM)
            self.dropped += len(self._buf)
            self._buf.clear()
        return lines
