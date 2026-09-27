"""Story flow, part 1: how the visitor tells the story (voice / sign / text) and confirms it.

States: method -> (voice_lang -> voice_ready -> listening -> transcribing | sign_lang -> signing | typing)
        -> confirm -> consent -> making -> showtime -> done       (part 2 in story_make.py)
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from yq.common import config
from yq.common.contracts import new_id
from yq.server import settings
from yq.server.flows.base import Flow, FlowError
from yq.server.flows.story_make import StoryMakeMixin

ES = "spa_Latn"


def sign_view(st: dict, eng) -> dict:
    """Normalize SignEngine.state() for the screen: candidates -> [{"text","prob","kind"}]."""
    cands = []
    for c in st.get("candidates") or []:
        if isinstance(c, dict):
            cands.append({"text": str(c.get("text") or c.get("value") or ""),
                          "prob": float(c.get("prob") or c.get("confidence") or 0.0), "kind": c.get("kind", "")})
        elif isinstance(c, (list, tuple)) and c:
            cands.append({"text": str(c[0]), "prob": float(c[1]) if len(c) > 1 else 0.0, "kind": ""})
    text = st.get("text")
    if text is None:
        try:
            text = eng.text()
        except Exception:
            text = ""
    letter = st.get("letter") if isinstance(st.get("letter"), dict) else None
    return {"hands_visible": bool(st.get("hands_visible")), "fps": st.get("fps"), "mode": st.get("mode"),
            "text": text or "", "buffer": st.get("buffer") or "", "candidates": cands[:5], "letter": letter,
            "error": st.get("error")}


class StoryFlow(StoryMakeMixin, Flow):
    kind = "story"
    first = "method"
    retry_map = {"listening": "voice_ready", "transcribing": "transcribing", "signing": "sign_lang",
                 "sign_lang": "sign_lang", "confirm": "confirm", "consent": "consent", "making": "making",
                 "showtime": "showtime"}

    def __init__(self, kiosk, method: str = ""):
        super().__init__(kiosk)
        self.method = method if method in ("voice", "sign", "text") else ""
        self.story_id = new_id("story")
        self.dir = Path(config.STORIES_DIR) / self.story_id
        self.text, self.lang, self.text_es, self.source, self.sign_lang = "", "", "", "", ""
        self.voice_lang = "auto"
        self.audio = None
        self.transcript: dict = {}
        self.engine = None
        self.rec_stop = threading.Event()
        self._tr_seq = 0
        self._sign_emit = 0.0
        self.immediate = {
            "stop_listening": lambda p: self.rec_stop.set(),
            "sign_accept": lambda p: self._sign("accept", int(p.get("index", 0))),
            "sign_backspace": lambda p: self._sign("backspace"),
            "sign_clear": lambda p: self._sign("clear"),
            "sign_space": lambda p: self._sign("space"),
            "sign_mode": lambda p: self._sign("set_mode", p.get("mode", "letters")),
            "consent": self._vote,
            "skip_narration": self._skip_narration,
        }
        self.init_make()

    def url(self, name: str) -> str:
        return "/files/stories/%s/%s" % (self.story_id, name)

    def on_cancel(self) -> None:
        self.rec_stop.set()
        self.sub.voice.stop_tts()
        self._stop_engine()

    def cleanup(self) -> None:
        self._stop_engine()
        self.cleanup_make()

    # -- choose how -----------------------------------------------------------------------
    def s_method(self):
        if self.method:
            m, self.method = self.method, ""           # preselected once (from the home screen)
        else:
            self.show("method", sign_ok=bool(self.sub.sign.usable()))
            _, p = self.wait("choose")
            m = p.get("method")
        return {"voice": "voice_lang", "sign": "sign_lang", "text": "typing"}.get(m, "method")

    # -- voice -------------------------------------------------------------------------------
    def s_voice_lang(self):
        self.show("voice_lang", languages_url="/api/languages?feature=asr", selected=self.voice_lang)
        name, p = self.wait("choose_lang", "back")
        if name == "back":
            return "method"
        self.voice_lang = p.get("code") or "auto"
        return "voice_ready"

    def s_voice_ready(self):
        self.show("voice_ready", lang=self.voice_lang, lang_name=self.sub.languages.name_es(self.voice_lang))
        name, _ = self.wait("record", "change_lang", "back")
        return {"record": "listening", "change_lang": "voice_lang"}.get(name, "method")

    def s_listening(self):
        self.rec_stop.clear()
        self.show("listening", max_s=float(settings.RECORD_MAX_S), lang_name=self.sub.languages.name_es(self.voice_lang))

        def level(v):
            self.k.touch()                              # talking counts as being here
            self.bus.emit("level", min_interval=0.06, level=round(float(v), 3))

        audio = self.call(self.sub.voice.record, on_level=level, max_s=float(settings.RECORD_MAX_S),
                          stop_event=self.rec_stop, timeout=float(settings.RECORD_MAX_S) + 20,
                          on_cancel=self.rec_stop.set)
        if audio is None or len(audio) < 16000 * 0.4:
            raise FlowError("No te escuché. Acércate al micrófono y habla un poco más fuerte.", "err_no_audio",
                            retry="voice_ready")
        self.audio = audio
        return "transcribing"

    def s_transcribing(self):
        self.show("transcribing", lang_name=self.sub.languages.name_es(self.voice_lang))
        t = self.call(self.sub.voice.transcribe, self.audio, self.voice_lang,
                      timeout=float(settings.TRANSCRIBE_TIMEOUT))
        if not (t or {}).get("text", "").strip():
            raise FlowError("No pude entender lo que dijiste. ¿Lo intentamos otra vez?", "err_no_speech",
                            retry="voice_ready")
        self.transcript = t
        self.text, self.lang, self.source = t["text"].strip(), t.get("lang") or ES, "voice"
        return "confirm"

    # -- sign language -----------------------------------------------------------------------
    def _sign_langs(self) -> list:
        try:
            return self.sub.sign.available()
        except Exception:
            return []

    def s_sign_lang(self):
        langs = self._sign_langs()
        usable = [s["code"] for s in langs if s.get("letters") or s.get("words")]
        if not usable:
            raise FlowError("La cámara de señas no está lista. Prueba con tu voz o escribiendo.", "err_sign_off",
                            retry="method")
        self.show("sign_lang", sign_langs=[dict(s, ready=s["code"] in usable) for s in langs])
        while True:
            name, p = self.wait("choose_sign", "back")
            if name == "back":
                return "method"
            if p.get("code") in usable:
                self.sign_lang = p["code"]
                return "signing"

    def s_signing(self):
        self._stop_engine()
        self.engine = self.sub.sign.engine(self.sign_lang, on_token=lambda tok: self.k.touch(),
                                           on_status=lambda st: self.bus.emit("sign_status", min_interval=0.3,
                                                                               status=st))
        self.engine.start()
        info = next((s for s in self._sign_langs() if s.get("code") == self.sign_lang), {})
        self.show("signing", sign_lang=self.sign_lang, sign_name=info.get("name_es", self.sign_lang),
                  letters=bool(info.get("letters", True)), words=int(info.get("words") or 0),
                  preview="/sign/preview.mjpeg", can_space=hasattr(self.engine, "space"))
        while True:
            name, _ = self.wait("sign_done", "back", poll=self._sign_poll)
            if name == "back":
                self._stop_engine()
                return "sign_lang"
            text = (self.engine.text() or "").strip()
            if text:
                break
            self.toast("Todavía no hay texto. Seña letras o palabras y tócalas.", "toast_sign_empty")
        self._stop_engine()
        self.text, self.source = text, "sign"
        self.lang = self.sub.sign.spoken_lang(self.sign_lang)
        return "confirm"

    def _sign_poll(self, force: bool = False) -> None:
        eng = self.engine
        if eng is None or (not force and time.time() - self._sign_emit < 0.2):
            return
        self._sign_emit = time.time()
        try:
            st = dict(eng.state())
        except Exception:
            return
        if st.get("hands_visible"):
            self.k.touch()                              # signing counts as being here
        self.bus.emit("sign", **sign_view(st, eng))

    def _sign(self, method: str, *args) -> None:
        eng = self.engine
        fn = getattr(eng, method, None) if eng is not None else None
        if fn is None:
            return
        fn(*args)
        self._sign_poll(force=True)

    def _stop_engine(self) -> None:
        eng, self.engine = self.engine, None
        if eng is not None:
            try:
                eng.stop()
            except Exception:
                pass

    # -- typing --------------------------------------------------------------------------------
    def s_typing(self):
        self.show("typing", text=self.text if self.source == "text" else "", lang=self.lang or ES,
                  lang_name=self.sub.languages.name_es(self.lang or ES), languages_url="/api/languages?feature=")
        name, p = self.wait("submit_text", "back")
        if name == "back":
            return "method"
        text = (p.get("text") or "").strip()
        if not text:
            return "typing"
        self.text, self.source, self.lang = text[:4000], "text", p.get("lang") or ES
        return "confirm"

    # -- confirm ----------------------------------------------------------------------------------
    def s_confirm(self):
        cands = [[c, round(float(p), 3), self.sub.languages.name_es(c)]
                 for c, p in (self.transcript.get("lang_candidates") or [])] if self.source == "voice" else []
        needs_tr = self.lang != ES
        self.text_es = "" if needs_tr else self.text
        self.show("confirm", text=self.text, lang=self.lang, lang_name=self.sub.languages.name_es(self.lang),
                  source=self.source, sign_lang=self.sign_lang, candidates=cands,
                  confidence=self.transcript.get("confidence") if self.source == "voice" else None,
                  text_es=self.text_es, translating=needs_tr)
        if needs_tr:
            self._translate_async()
        while True:
            name, p = self.wait("confirm", "edit_text", "retell", "relang")
            if name == "confirm":
                self._finish_translation()
                return "consent"
            if name == "edit_text" and (p.get("text") or "").strip():
                self.text = p["text"].strip()[:4000]
                self.text_es = "" if self.lang != ES else self.text
                self.update(text=self.text, text_es=self.text_es, translating=self.lang != ES)
                if self.lang != ES:
                    self._translate_async()
            elif name == "retell":
                return {"voice": "voice_ready", "sign": "signing", "text": "typing"}.get(self.source, "method")
            elif name == "relang" and p.get("code") and self.source == "voice":
                self.voice_lang = p["code"]
                return "transcribing"

    def _translate_async(self) -> None:
        self._tr_seq += 1
        seq, text, lang = self._tr_seq, self.text, self.lang

        def work():
            try:
                es = self.sub.voice.translate(text, lang, ES) or ""
            except Exception:
                es = ""
            if seq == self._tr_seq and not self.cancel_event.is_set():
                self.text_es = es if es.strip() != text.strip() or lang == ES else ""
                self.update(text_es=self.text_es, translating=False, translation_failed=not self.text_es)

        self._tr_thread = threading.Thread(target=work, name="translate", daemon=True)
        self._tr_thread.start()

    def _finish_translation(self) -> None:
        th = getattr(self, "_tr_thread", None)
        if th is not None and th.is_alive():
            self.update(translating=True)
            self.call(th.join, float(settings.TRANSLATE_TIMEOUT))     # gives up quietly: no Spanish line
