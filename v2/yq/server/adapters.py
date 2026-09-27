"""Thin adapters between the kiosk flows and the other domains (CONTRACTS.md §6).

Each subsystem is resolved once, lazily: the real module when it imports, else the
UI mock from mocks.py (and the reason is kept for /status). YQ_UI_MOCKS=all (or a
list) forces the UI mocks, which is what the UI tests do. Real modules still use
their OWN mocks under YQ_MOCK / YQ_MOCK_<NAME>; the adapter does not care.
"""
from __future__ import annotations

import dataclasses
import importlib
import logging
import threading
from typing import Any, Optional

from yq.common import config
from yq.server import mocks, settings

log = logging.getLogger("yq.server.adapters")
_QR_LOCK = threading.Lock()


def as_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        from yq.common.contracts import to_dict
        return to_dict(obj)
    return obj


def _imp(name: str):
    """Import a module; returns (module, None) or (None, 'ErrorType: message')."""
    try:
        return importlib.import_module(name), None
    except Exception as e:                       # ImportError, SyntaxError, missing deps...
        return None, "%s: %s" % (type(e).__name__, str(e)[:160])


class Voice:
    """record / transcribe / translate / say / stop_tts / synthesize, each from yq.voice when present."""

    def __init__(self):
        self._mock = mocks.MockVoice()
        self.parts: dict = {}
        forced = settings.ui_mocked("voice")
        for part in ("capture", "asr", "tts", "translate"):
            mod, err = (None, "forced UI mock") if forced else _imp("yq.voice." + part)
            self.parts[part] = (mod, err)

    def _m(self, part):
        return self.parts[part][0]

    @property
    def mode(self) -> str:
        real = [p for p, (m, _) in self.parts.items() if m is not None]
        if not real:
            return "mock:ui"
        flags = [f for f in ("mic", "speaker", "mac") if config.mock(f)]
        base = "yq.voice" if len(real) == 4 else "yq.voice parcial (%s)" % ",".join(real)
        return base + (" [mock %s]" % ",".join(flags) if flags else "")

    def record(self, on_level=None, max_s: float = 90.0, stop_event=None):
        m = self._m("capture")
        if m is None:
            return self._mock.record(on_level=on_level, max_s=max_s, stop_event=stop_event)
        return m.Recorder().record(on_level=on_level, max_s=max_s, stop_event=stop_event)

    def transcribe(self, audio, lang: str = "auto") -> dict:
        m = self._m("asr")
        t = self._mock.transcribe(audio, lang) if m is None else m.transcribe(audio, lang=lang)
        return as_dict(t)

    def translate(self, text: str, src: str, tgt: str) -> str:
        if not text or src == tgt:
            return text
        m = self._m("translate")
        return self._mock.translate(text, src, tgt) if m is None else m.translate(text, src, tgt)

    def say(self, text: str, lang: str, wait: bool = False) -> bool:
        m = self._m("tts")
        return self._mock.say(text, lang, wait=wait) if m is None else bool(m.say(text, lang, wait=wait))

    def stop_tts(self) -> None:
        m = self._m("tts")
        try:
            (self._mock.stop_tts() if m is None else m.stop())
        except Exception:
            log.exception("tts stop failed")

    def synthesize(self, text: str, lang: str) -> Optional[bytes]:
        """WAV bytes for the hologram package (not in the contract yet: see docs/ui.md)."""
        m = self._m("tts")
        if m is None:
            return self._mock.synthesize(text, lang)
        fn = getattr(m, "synthesize", None)
        if fn is not None:
            return to_wav(fn(text, lang))
        try:                                     # Mac worker /tts (CONTRACTS §3.1) returns audio/wav
            from yq.common.macclient import client
            r = client()._request("POST", "/tts", json={"text": text, "lang": lang}, timeout=60.0)
            return r.content if r.ok else None
        except Exception:
            return None


def to_wav(x) -> Optional[bytes]:
    """WAV bytes from bytes, or from yq.voice.tts.synthesize's (samples, rate[, engine]) tuple."""
    if x is None or isinstance(x, (bytes, bytearray)):
        return bytes(x) if x else None
    import io
    import wave
    import numpy as np
    audio, rate = x[0], int(x[1])
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class Languages:
    PERU = {"spa", "quy", "quz", "que", "qub", "qud", "quf", "qug", "qvc", "qve", "qvi", "qvw", "qwh", "qxn",
            "ayr", "aym", "cni", "agr", "shp", "ame", "cbu", "huu", "mcb", "ote", "prq", "jiv", "kaq"}

    def __init__(self):
        mod, err = (None, "forced UI mock") if settings.ui_mocked("languages") else _imp("yq.common.languages")
        self.mod, self.err = mod, err
        self._impl = mod if mod is not None else mocks.MockLanguages()
        self.mode = "yq.common.languages" if mod is not None else "mock:ui"

    def get(self, code: str) -> Optional[dict]:
        try:
            lang = self._impl.get(code)
        except Exception:
            lang = None
        return as_dict(lang) if lang is not None else None

    def name_es(self, code: str) -> str:
        if code in ("auto", "", None):
            return "Automático"
        d = self.get(code) or {}
        return d.get("name_es") or d.get("name") or code

    def has_tts(self, code: str) -> bool:
        d = self.get(code) or {}
        return bool(d.get("tts"))

    def ui_list(self, feature: str = "asr") -> list:
        """Normalized list for the kiosk: Peruvian languages first, then by popularity."""
        try:
            rows = [as_dict(r) for r in self._impl.ui_list(feature)]
        except Exception:
            log.exception("languages.ui_list failed; using the UI list")
            rows = mocks.MockLanguages().ui_list(feature)
        out = []
        for r in rows:
            code = r.get("code") or ""
            iso = (r.get("iso639_3") or code.split("_")[0]).lower()
            region = str(r.get("region") or "")
            peru = r["peru"] if "peru" in r else (iso in self.PERU or region.upper() in ("PE", "PERÚ", "PERU"))
            out.append({"code": code, "name": r.get("name_es") or r.get("name") or code,
                        "native": r.get("name_native") or "", "en": r.get("name_en") or "",
                        "peru": bool(peru), "popular": int(r.get("popular") or 0)})
        # `popular` is a display rank (VOICE): 1 = first, 0 = unranked (after the ranked ones, by name)
        out.sort(key=lambda d: (not d["peru"], d["popular"] == 0, d["popular"], d["name"]))
        return out


class Sign:
    SPOKEN = {"ase": "eng_Latn", "prl": "spa_Latn", "ils": "eng_Latn"}

    def __init__(self):
        mod, err = (None, "forced UI mock") if settings.ui_mocked("sign") else _imp("yq.sign")
        if mod is not None and not (hasattr(mod, "available") and hasattr(mod, "SignEngine")):
            mod, err = None, "yq.sign has no available()/SignEngine yet"
        self.mod, self.err = mod, err
        self._impl = mod if mod is not None else mocks.MockSign()
        self.mode = ("yq.sign" + (" [mock camera]" if config.mock("sign_camera") else "")) if mod else "mock:ui"

    def available(self) -> list:
        return [as_dict(x) for x in self._impl.available()]

    def usable(self) -> list:
        """Sign languages with at least one installed model (letters or words)."""
        return [s for s in self.available() if s.get("letters") or s.get("words")]

    def engine(self, sign_lang: str, on_token=None, on_status=None):
        return self._impl.SignEngine(sign_lang, camera=None, on_token=on_token, on_status=on_status)

    def spoken_lang(self, sign_lang: str) -> str:
        for s in self.available():
            if s.get("code") == sign_lang and s.get("spoken"):
                return s["spoken"]
        return self.SPOKEN.get(sign_lang, "spa_Latn")


class Art:
    def __init__(self):
        mod, err = (None, "forced UI mock") if settings.ui_mocked("art") else _imp("yq.art.drawing")
        if mod is not None and not hasattr(mod, "make_drawing"):
            mod, err = None, "yq.art.drawing has no make_drawing yet"
        self.mod, self.err = mod, err
        self._impl = mod if mod is not None else mocks.MockArt()
        self.mode = "yq.art" if mod is not None else "mock:ui"

    def make_drawing(self, story, out_dir, on_progress=None, published: bool = True) -> dict:
        """published=False (visitor declined): the printed QR must point to the general gallery page.
        Passed as a keyword when ART accepts it; otherwise ART's QR format is set to "{base}" for this call."""
        import inspect
        fn = self._impl.make_drawing
        try:
            takes = "published" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            takes = False
        if takes:
            return as_dict(fn(story, out_dir, on_progress=on_progress, published=published))
        if published or self.mod is None:
            return as_dict(fn(story, out_dir, on_progress=on_progress))
        art_settings, _ = _imp("yq.art.settings")
        with _QR_LOCK:                                   # one drawing at a time while the format is changed
            old = getattr(art_settings, "QR_URL_FORMAT", None) if art_settings else None
            if old is not None:
                art_settings.QR_URL_FORMAT = "{base}"
            try:
                d = as_dict(fn(story, out_dir, on_progress=on_progress))
            finally:
                if old is not None:
                    art_settings.QR_URL_FORMAT = old
        if d and d.get("qr_url") and d["qr_url"].rstrip("/") != config.PUBLIC_BASE_URL.rstrip("/"):
            log.warning("ART printed a story QR for a private story: %s", d.get("qr_url"))
        return d


class Box:
    def __init__(self):
        mod, err = (None, "forced UI mock") if settings.ui_mocked("box") else _imp("yq.box.scan")
        if mod is not None and not (hasattr(mod, "preflight") and hasattr(mod, "run_scan")):
            mod, err = None, "yq.box.scan has no preflight()/run_scan() yet"
        self.mod, self.err = mod, err
        self._impl = mod if mod is not None else mocks.MockBox()
        self.mode = ("yq.box" + (" [simulada]" if config.mock("box") else "")) if mod is not None else "mock:ui"
        self.lock = threading.Lock()             # one scan at a time, also across flows

    def preflight(self) -> dict:
        return as_dict(self._impl.preflight())

    def run_scan(self, req, on_progress=None, cancel_event=None) -> dict:
        return as_dict(self._impl.run_scan(req, on_progress=on_progress, cancel_event=cancel_event))


def printer():
    if config.mock("printer") or settings.ui_mocked("printer"):
        from yq.printer.mock import MockPrinterClient
        return MockPrinterClient()
    from yq.printer.client import PrinterClient
    return PrinterClient()


def hologram():
    if config.mock("hologram") or settings.ui_mocked("hologram"):
        from yq.hologram.mock import MockHologramClient
        return MockHologramClient()
    from yq.hologram.client import HologramClient
    return HologramClient()


class Subsystems:
    """Everything a flow may call, resolved once per kiosk."""

    def __init__(self):
        from yq.server.camera import make_camera
        self.voice = Voice()
        self.languages = Languages()
        self.sign = Sign()
        self.art = Art()
        self.box = Box()
        self.camera = make_camera()
        self.printer = printer()
        self.hologram = hologram()

    def modes(self) -> dict:
        errs = {"voice": {p: e for p, (m, e) in self.voice.parts.items() if e},
                "languages": self.languages.err, "sign": self.sign.err, "art": self.art.err, "box": self.box.err}
        return {"voice": self.voice.mode, "languages": self.languages.mode, "sign": self.sign.mode,
                "art": self.art.mode, "box": self.box.mode, "camera": self.camera.mode,
                "printer": getattr(self.printer, "source", "?"), "hologram": getattr(self.hologram, "source", "?"),
                "errors": {k: v for k, v in errs.items() if v}}
