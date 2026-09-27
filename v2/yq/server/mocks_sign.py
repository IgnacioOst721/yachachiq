"""UI mock of yq.sign, shaped like the real SignEngine (yq/sign/engine.py):

letters mode: signed letters pile up in `buffer`; `candidates` are word completions
[{"text","prob","kind"}]; accept(i) commits one; a finished word commits by itself.
words mode: each recognised sign is appended; `candidates` are alternatives for the
last word and accept(i) swaps it. state() also has "text" (everything) and "letter".
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from yq.server import mock_assets, settings

SIGN_LANGS = [
    {"code": "prl", "name_es": "Lengua de Señas Peruana (LSP)", "letters": True, "words": 40, "spoken": "spa_Latn"},
    {"code": "ase", "name_es": "Lengua de señas americana (ASL)", "letters": True, "words": 60, "spoken": "eng_Latn"},
    {"code": "ils", "name_es": "Señas Internacionales (IS)", "letters": False, "words": 30, "spoken": "eng_Latn"},
]
SCRIPT = ["hola", "condor", "vuela", "alto"]
LEXICON = ["hola", "hoja", "hombre", "condor", "con", "cono", "comer", "vuela", "vuelo", "verde", "alto", "algo", "alma"]
GLOSSES = [("CONDOR", "cóndor", ["CASA", "COMER"]), ("VOLAR", "vuela", ["VER", "VOLVER"]),
           ("MONTAÑA", "montaña", ["MAÑANA", "MONTE"]), ("LLAMA", "llama", ["LLAVE", "LLUVIA"])]


class MockSignEngine:
    source = "mock:ui"

    def __init__(self, sign_lang: str, camera=None, on_token=None, on_status=None):
        info = next((s for s in SIGN_LANGS if s["code"] == sign_lang), SIGN_LANGS[0])
        self.sign_lang, self.on_token, self.on_status = sign_lang, on_token, on_status
        self.mode = "letters" if info["letters"] else "words"
        self._words: list = []
        self._buf = ""
        self._cands: list = []
        self._step = 0
        self._jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fps = 0.0

    # -- contract ------------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        threading.Thread(target=self._run, name="mock-sign", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def set_mode(self, mode: str) -> None:
        if mode in ("letters", "words"):
            with self._lock:
                self.mode, self._buf, self._cands, self._step = mode, "", [], 0

    def latest_jpeg(self) -> Optional[bytes]:
        return self._jpeg

    def text(self) -> str:
        return " ".join(self._words + ([self._buf] if self._buf else []))

    def state(self) -> dict:
        with self._lock:
            letter = {"current": self._buf[-1], "conf": 0.82, "progress": 0.6} if self._buf else None
            return {"hands_visible": True, "fps": round(self._fps, 1), "mode": self.mode, "sign_lang": self.sign_lang,
                    "text": self.text(), "buffer": self._buf, "letter": letter,
                    "candidates": [{"text": t, "prob": p, "kind": "word"} for t, p in self._cands]}

    def accept(self, index: int = 0) -> Optional[str]:
        with self._lock:
            if not self._cands:
                return None
            word = self._cands[min(index, len(self._cands) - 1)][0]
            if self.mode == "letters":
                self._words.append(word)
                self._buf, self._step = "", self._step + 1
            elif self._words:
                self._words[-1] = word
            self._cands = []
        if self.on_token:
            self.on_token({"kind": "word", "value": word, "text": word})
        return word

    def space(self) -> None:
        """Not in the real engine (yet): commits the letters signed so far as a word."""
        with self._lock:
            if self._buf:
                self._words.append(self._buf)
                self._buf, self._cands, self._step = "", [], self._step + 1

    def backspace(self) -> None:
        with self._lock:
            if self._buf:
                self._buf = self._buf[:-1]
            elif self._words:
                self._words.pop()
            self._cands = self._completions() if self._buf else []

    def clear(self) -> None:
        with self._lock:
            self._words, self._buf, self._cands = [], "", []

    # -- simulation ----------------------------------------------------------------------------
    def _completions(self) -> list:
        hits = [w for w in LEXICON if w.startswith(self._buf)][:3]
        probs = [0.72, 0.18, 0.07]
        return [(w, probs[i]) for i, w in enumerate(hits)]

    def _tick(self) -> None:
        with self._lock:
            if self.mode == "letters":
                target = SCRIPT[self._step % len(SCRIPT)]
                if len(self._buf) < len(target):
                    self._buf += target[len(self._buf)]
                    self._cands = self._completions()
                else:                                    # hand down: the word closes by itself
                    self._words.append(self._buf)
                    self._buf, self._cands, self._step = "", [], self._step + 1
            else:
                gloss, text, alts = GLOSSES[self._step % len(GLOSSES)]
                self._words.append(text)
                self._cands = [(text, 0.74)] + [(a.lower(), 0.12) for a in alts]
                self._step += 1

    def _run(self) -> None:
        t0, frames, last = time.time(), 0, time.time()
        while not self._stop.is_set():
            now = time.time()
            period = (0.9 if self.mode == "letters" else 2.2) * float(settings.MOCK_SPEED)
            if now - last >= period:
                last = now
                self._tick()
            self._jpeg = mock_assets.sign_frame(now - t0, self._buf[-1:] if self._buf else "")
            frames += 1
            self._fps = frames / max(0.001, now - t0)
            self._stop.wait(1 / 12.0)


class MockSign:
    source = "mock:ui"
    SignEngine = MockSignEngine

    def available(self) -> list:
        return [dict(s) for s in SIGN_LANGS]
