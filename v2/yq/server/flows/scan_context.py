"""Scan flow, optional step "¿Dónde lo encontraron?" (CONTRACTS.md §9).

The visitor may type (on-screen keyboard) or dictate (yq.voice Recorder + transcribe) where
the object was found, and/or tap a region chip. It becomes ScanRequest.context
{"found_where", "region_hint", "notes", "lang"}: a CLUE for identification, never proof.
"""
from __future__ import annotations

import logging
import threading

from yq.server.flows.base import Cancelled

log = logging.getLogger("yq.server.scan")

REGIONS = ("costa_norte", "costa_central", "costa_sur", "sierra_norte", "sierra_central", "sierra_sur",
           "altiplano", "selva", "lima", "otro_pais", "no_se")
MAX_CHARS = 200
DICTATE_MAX_S = 25.0
UI_LANGS = {"es": "spa_Latn", "en": "eng_Latn", "qu": "auto"}   # dictation language from the UI language
TYPED_LANGS = {"es": "spa_Latn", "en": "eng_Latn", "qu": "quy_Latn"}


class ScanContextMixin:
    def init_context(self) -> None:
        self.context: dict = {}
        self._ctx = {"found_where": "", "region_hint": "", "lang": ""}
        self.dictate_stop = threading.Event()
        self.immediate["stop_dictation"] = lambda p: self.dictate_stop.set()

    def build_context(self) -> dict:
        text = (self._ctx.get("found_where") or "").strip()[:MAX_CHARS]
        region = self._ctx.get("region_hint") if self._ctx.get("region_hint") in REGIONS else ""
        if not text and not region:
            return {}
        return {"found_where": text, "region_hint": region, "notes": "", "lang": self._ctx.get("lang") or "spa_Latn"}

    def _take(self, p: dict) -> None:
        if "found_where" in p:
            self._ctx["found_where"] = str(p.get("found_where") or "")[:MAX_CHARS]
            if p.get("lang"):                            # typed: the UI language is our best guess
                self._ctx["lang"] = TYPED_LANGS.get(p["lang"], p["lang"])
        if "region_hint" in p:
            r = p.get("region_hint") or ""
            self._ctx["region_hint"] = r if r in REGIONS else ""

    def s_context(self):
        self.show("context", regions=list(REGIONS), max_chars=MAX_CHARS, recording=False, transcribing=False,
                  **{k: self._ctx.get(k, "") for k in ("found_where", "region_hint")})
        while True:
            name, p = self.wait("set_context", "dictate", "continue", "skip", "back")
            if name == "back":
                return "intro"
            if name == "skip":
                self._ctx = {"found_where": "", "region_hint": "", "lang": ""}
                self.context = {}
                return "preflight"
            self._take(p)
            if name == "set_context":
                self.update(**{k: self._ctx[k] for k in ("found_where", "region_hint")})
            elif name == "dictate":
                self._dictate(UI_LANGS.get(p.get("ui_lang") or "es", "auto"))
            elif name == "continue":
                self.context = self.build_context()
                return "preflight"

    def _dictate(self, lang: str) -> None:
        """Record + transcribe; a failure is a toast, never an error screen (the step is optional)."""
        self.dictate_stop.clear()
        self.update(recording=True, transcribing=False)

        def level(v):
            self.k.touch()
            self.bus.emit("level", min_interval=0.06, level=round(float(v), 3))
        try:
            audio = self.call(self.sub.voice.record, on_level=level, max_s=DICTATE_MAX_S, stop_event=self.dictate_stop,
                              timeout=DICTATE_MAX_S + 15, on_cancel=self.dictate_stop.set)
            self.update(recording=False, transcribing=True)
            if audio is None or len(audio) < 16000 * 0.4:
                raise ValueError("no audio")
            t = self.call(self.sub.voice.transcribe, audio, lang, timeout=120) or {}
            text = (t.get("text") or "").strip()
            if not text:
                raise ValueError("empty transcript")
            self._ctx["found_where"] = text[:MAX_CHARS]
            self._ctx["lang"] = t.get("lang") or (lang if lang != "auto" else "spa_Latn")
            self.update(recording=False, transcribing=False, found_where=self._ctx["found_where"], dictated=True)
        except Cancelled:
            raise
        except Exception as e:
            log.info("dictation failed: %s", e)
            self.update(recording=False, transcribing=False)
            self.toast("No te escuché bien. Puedes intentarlo otra vez o escribirlo.", "ctx_no_audio")
