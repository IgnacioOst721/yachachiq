import json
import re
from pathlib import Path

import pytest

from yq.box import protocol as P
from yq.box import settings as S

FW = Path(__file__).resolve().parents[2] / "firmware" / "box_esp32"


def test_encode_request_is_one_compact_line():
    line = P.encode_request(7, "rotate_to", deg=12.5, wrap=None)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert json.loads(line) == {"id": 7, "cmd": "rotate_to", "deg": 12.5}   # None params dropped


def test_encode_rejects_unknown_and_long():
    with pytest.raises(P.ProtocolError):
        P.encode_request(1, "self_destruct")
    with pytest.raises(P.ProtocolError):
        P.encode_request(1, "config_set", note="x" * 600)


def test_decode_reply_event_and_garbage():
    r = P.decode_line(b'{"id":3,"ok":true,"deg":1.5}\r\n')
    assert r["_kind"] == "reply" and r["deg"] == 1.5
    e = P.decode_line('{"event":"door","door":"front","closed":false}')
    assert e["_kind"] == "event" and e["closed"] is False
    for bad in (b"", b"ets Jun  8 2016 00:22:57", b"{not json", b"[1,2]", b'{"x":1}'):
        with pytest.raises(P.ProtocolError):
            P.decode_line(bad)


def test_raise_for_reply():
    assert P.raise_for_reply({"id": 1, "ok": True})["ok"]
    with pytest.raises(P.BoxError) as ei:
        P.raise_for_reply({"id": 1, "ok": False, "error": "interlock", "msg": "door open"})
    assert ei.value.code == "interlock" and "puerta" in ei.value.message_es


def test_line_buffer_splits_and_drops_noise():
    lb = P.LineBuffer(max_line=64)
    assert lb.feed(b'{"a":1}\n{"b"') == [b'{"a":1}']
    assert lb.feed(b':2}\r\n\n') == [b'{"b":2}']
    assert lb.feed(b"x" * 100) == [] and lb.dropped == 100     # boot ROM noise without newline


def test_protocol_md_documents_every_command_event_and_error():
    doc = (FW / "PROTOCOL.md").read_text()
    for name in P.COMMANDS + P.EVENTS + P.ERRORS:
        assert "`%s`" % name in doc, name


def test_firmware_dispatches_every_command():
    src = (FW / "src" / "commands.cpp").read_text()
    for cmd in P.COMMANDS:
        assert '"%s"' % cmd in src, cmd


def test_settings_mirror_firmware_pins_and_defaults():
    pins = (FW / "include" / "pins.h").read_text()
    define = dict(re.findall(r"#define PIN_(\w+) (\d+)", pins))
    assert int(define["STEP"]) == S.PINS["step"] and int(define["DIR"]) == S.PINS["dir"]
    assert int(define["EN"]) == S.PINS["en"] and int(define["UV"]) == S.PINS["uv"]
    assert int(define["HALOGEN"]) == S.PINS["halogen"] and int(define["COB"]) == S.PINS["cob"]
    assert int(define["FAN"]) == S.PINS["fan"] and int(define["HX_SCK"]) == S.PINS["hx_sck"]
    assert int(define["REED_FRONT"]) == S.PINS["reed_front"] and int(define["REED_SHUTTER"]) == S.PINS["reed_shutter"]
    rake = [int(x) for x in re.search(r"RAKE_PINS\[NUM_RAKE\] = \{([^}]*)\}", pins).group(1).split(",")]
    assert rake == S.PINS["rake"]
    cfg = (FW / "include" / "boxconfig.h").read_text()
    on = [int(x) for x in re.search(r"max_on_ms\[K_COUNT\] = \{([^}]*)\}", cfg).group(1).split(",")]
    assert on == [S.FIRMWARE_DEFAULTS["max_on_ms"][k] for k in ("rake", "uv", "halogen", "cob", "fan")]
    hard = [int(x) for x in re.search(r"HARD_MAX_ON_MS\[K_COUNT\] = \{([^}]*)\}", cfg).group(1).split(",")]
    assert hard == [S.HARD_MAX_ON_MS[k] for k in ("rake", "uv", "halogen", "cob", "fan")]
    assert S.STEPS_PER_REV == 44800


def test_light_geometry_is_unit_vectors_from_cad():
    lights = S.light_geometry()
    assert [L["index"] for L in lights] == list(range(1, 9))
    for L in lights:
        assert abs(sum(c * c for c in L["direction"]) - 1) < 1e-4
        assert 15 < L["elevation_deg"] < 40          # raking: low but above the platter
